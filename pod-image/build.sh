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
# Usage:
#   ./build.sh                     # refresh vendored deps + build all three
#   ./build.sh --push              # ...and push to the registry
#   ./build.sh base                # build only one (also: train, cache)
#   ./build.sh base train --push
#   GPT_GOLF_DIR=/path ./build.sh  # non-default gpt-golf checkout
#   REGISTRY=ghcr.io/me ./build.sh # non-default registry/owner
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPT_GOLF_DIR="${GPT_GOLF_DIR:-$HERE/../../gpt-golf}"
REGISTRY="${REGISTRY:-ghcr.io/sysreq}"
BASE_IMAGE="$REGISTRY/gpt-caddy-base:latest"
TRAIN_IMAGE="$REGISTRY/gpt-caddy-single-train:latest"
CACHE_IMAGE="$REGISTRY/gpt-caddy-cache-server:latest"

PUSH=0
targets=()
for arg in "$@"; do
    case "$arg" in
        --push)            PUSH=1 ;;
        base|train|cache)  targets+=("$arg") ;;
        *) echo "unknown arg: $arg (expected: base|train|cache|--push)" >&2; exit 2 ;;
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

build() {
    local file="$1" tag="$2"
    echo "==> Building $tag ($file)"
    docker build -f "$HERE/$file" -t "$tag" "$HERE"
    if [ "$PUSH" -eq 1 ]; then echo "==> Pushing $tag"; docker push "$tag"; fi
}

for t in "${targets[@]}"; do
    case "$t" in
        base)  build Dockerfile.base  "$BASE_IMAGE" ;;
        train) refresh_deps; build Dockerfile "$TRAIN_IMAGE" ;;
        cache) build Dockerfile.redis "$CACHE_IMAGE" ;;
    esac
done

echo "==> Done."
