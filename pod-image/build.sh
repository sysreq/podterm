#!/usr/bin/env bash
# Build (and optionally push) the RunPod images for PodTerm's training pods.
#
# Three images, layered base -> {train, cache-server}:
#   ghcr.io/sysreq/gpt-caddy-base:latest          (Dockerfile.base)   torch/CUDA venv; rebuilt rarely
#   ghcr.io/sysreq/gpt-caddy-single-train:latest  (Dockerfile)        clone-gpt-golf + train entrypoint
#   ghcr.io/sysreq/gpt-caddy-cache-server:latest  (Dockerfile.redis)  shared torch.compile (Inductor) cache
#
# The training image bakes gpt-golf's *dependency closure* so the pod's runtime
# `uv sync` is a no-op — but the training CODE itself is git-cloned at boot
# (bootstrap.sh). That closure is defined by gpt-golf's pyproject.toml + uv.lock,
# which we VENDOR here (./pyproject.toml, ./uv.lock — NOT PodTerm's own). This
# script refreshes the vendored copies from the gpt-golf checkout before building
# so the baked venv can never silently drift from gpt-golf's lock.
#
# --push recompresses each layer to zstd via skopeo before uploading (smaller blobs =>
# faster cold pulls on ephemeral pods). Needs skopeo installed (apt install skopeo);
# falls back to plain `docker push` (gzip) if it's missing. Tune with ZSTD_LEVEL.
#
# Usage:
#   ./build.sh                     # refresh vendored deps + build all three
#   ./build.sh --push              # ...and push (zstd via skopeo) to the registry
#   ./build.sh base                # build only one (also: train, cache)
#   ./build.sh base train --push
#   GPT_GOLF_DIR=/path ./build.sh  # non-default gpt-golf checkout
#   REGISTRY=ghcr.io/me ./build.sh # non-default registry/owner
#   ZSTD_LEVEL=19 ./build.sh --push # compress harder (smaller image, slower push)
#   ./build.sh sizes               # report remote (registry) pull size of all three — no build
#   ./build.sh sizes base          # ...just one
#
# After a zstd push the script verifies the uploaded layers are actually zstd, then
# (on --push) prints the resulting remote sizes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPT_GOLF_DIR="${GPT_GOLF_DIR:-$HERE/../../gpt-golf}"
REGISTRY="${REGISTRY:-ghcr.io/sysreq}"
BASE_IMAGE="$REGISTRY/gpt-caddy-base:latest"
TRAIN_IMAGE="$REGISTRY/gpt-caddy-single-train:latest"
CACHE_IMAGE="$REGISTRY/gpt-caddy-cache-server:latest"

tag_for() {
    case "$1" in
        base)  echo "$BASE_IMAGE" ;;
        train) echo "$TRAIN_IMAGE" ;;
        cache) echo "$CACHE_IMAGE" ;;
    esac
}

PUSH=0 SIZES=0
targets=()
for arg in "$@"; do
    case "$arg" in
        --push)            PUSH=1 ;;
        sizes|--sizes)     SIZES=1 ;;
        base|train|cache)  targets+=("$arg") ;;
        *) echo "unknown arg: $arg (expected: base|train|cache|sizes|--push)" >&2; exit 2 ;;
    esac
done
[ "${#targets[@]}" -eq 0 ] && targets=(base train cache)

# Refresh the vendored gpt-golf manifest so the baked venv matches gpt-golf's lock.
# Prefer the live checkout; fall back to the committed vendored copy; fail loud if neither.
refresh_deps() {
    if [ -f "$GPT_GOLF_DIR/pyproject.toml" ] && [ -f "$GPT_GOLF_DIR/uv.lock" ]; then
        echo "==> Refreshing vendored deps from $GPT_GOLF_DIR"
        cp "$GPT_GOLF_DIR/pyproject.toml" "$GPT_GOLF_DIR/uv.lock" "$HERE/"
    elif [ -f "$HERE/pyproject.toml" ] && [ -f "$HERE/uv.lock" ]; then
        echo "==> WARNING: no gpt-golf checkout at $GPT_GOLF_DIR; using committed vendored deps (may be stale)" >&2
    else
        echo "==> FATAL: no gpt-golf checkout at $GPT_GOLF_DIR and no vendored pyproject.toml/uv.lock in $HERE" >&2
        exit 1
    fi
}

# Push the just-built image. With skopeo, copy daemon -> registry recompressing layers
# to zstd (smaller blobs, faster cold pulls). Without it, plain `docker push` (gzip).
# --dest-force-compress-format guarantees every layer is emitted as zstd (no blob reuse).
# --insecure-policy skips signature-policy lookup (no /etc/containers/policy.json needed);
# fine for copying our own image to our own registry. Creds come from `docker login`.
push_image() {
    local tag="$1"
    if command -v skopeo >/dev/null 2>&1; then
        local args=(copy --insecure-policy --dest-compress-format zstd --dest-force-compress-format)
        [ -n "${ZSTD_LEVEL:-}" ] && args+=(--dest-compress-level "$ZSTD_LEVEL")
        args+=("docker-daemon:$tag" "docker://$tag")
        echo "==> Pushing $tag (zstd via skopeo${ZSTD_LEVEL:+ level=$ZSTD_LEVEL})"
        skopeo "${args[@]}"
        verify_zstd "$tag"
    else
        echo "==> skopeo not installed — pushing $tag with gzip (docker push). 'apt install skopeo' for zstd." >&2
        docker push "$tag"
    fi
}

# Confirm the just-pushed image's layers really are zstd (the format we asked for).
verify_zstd() {
    local tag="$1" n
    n=$(skopeo inspect --raw "docker://$tag" 2>/dev/null | grep -o 'tar+zstd' | wc -l | tr -d ' ') || n=0
    if [ "${n:-0}" -gt 0 ]; then
        echo "==> Verified $tag: $n zstd layer(s)"
    else
        echo "==> WARNING: $tag shows no 'tar+zstd' layers — uploaded as gzip?" >&2
    fi
}

# Human-readable bytes (IEC, e.g. 1.4GiB). Falls back to raw bytes without numfmt.
human() { numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1}B"; }

# Report each image's *compressed* size on the registry (layer blobs + config) — roughly
# what a cold pod downloads. Assumes single-arch manifests; flags anything else as "?".
report_sizes() {
    command -v skopeo >/dev/null 2>&1 || { echo "==> sizes needs skopeo" >&2; return 1; }
    local tag info
    printf '%-50s %12s  %s\n' "IMAGE" "PULL SIZE" "LAYERS"
    for tag in "$@"; do
        info=$(skopeo inspect --raw "docker://$tag" 2>/dev/null | python3 -c '
import json, sys
try: m = json.load(sys.stdin)
except Exception: print("err 0"); sys.exit()
L = m.get("layers")
if L: print(sum(x.get("size", 0) for x in L) + m.get("config", {}).get("size", 0), len(L))
elif m.get("manifests"): print("list", len(m["manifests"]))
else: print("err 0")
') || info="err 0"
        case "${info%% *}" in
            err)  printf '%-50s %12s\n'     "$tag" "(not pushed?)" ;;
            list) printf '%-50s %12s  %s\n' "$tag" "(manifest list)" "${info##* } arch" ;;
            *)    printf '%-50s %12s  %s\n' "$tag" "$(human "${info%% *}")" "${info##* }" ;;
        esac
    done
}

build() {
    local file="$1" tag="$2"
    echo "==> Building $tag ($file)"
    docker build -f "$HERE/$file" -t "$tag" "$HERE"
    if [ "$PUSH" -eq 1 ]; then push_image "$tag"; fi
}

# `sizes` mode: just query the registry, don't build anything.
if [ "$SIZES" -eq 1 ]; then
    size_tags=()
    for t in "${targets[@]}"; do size_tags+=("$(tag_for "$t")"); done
    report_sizes "${size_tags[@]}"
    exit 0
fi

for t in "${targets[@]}"; do
    case "$t" in
        base)  build Dockerfile.base  "$BASE_IMAGE" ;;
        train) refresh_deps; build Dockerfile "$TRAIN_IMAGE" ;;
        cache) build Dockerfile.redis "$CACHE_IMAGE" ;;
    esac
done

# After a push, summarize the resulting remote sizes.
if [ "$PUSH" -eq 1 ] && command -v skopeo >/dev/null 2>&1; then
    pushed_tags=()
    for t in "${targets[@]}"; do pushed_tags+=("$(tag_for "$t")"); done
    echo "==> Remote sizes:"
    report_sizes "${pushed_tags[@]}"
fi

echo "==> Done."
