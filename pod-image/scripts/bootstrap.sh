#!/usr/bin/env bash
set -euo pipefail

# Container entrypoint (baked at /usr/local/bin/bootstrap.sh). Takes an ephemeral RunPod pod
# from cold boot -> trained model -> self-teardown, unattended. Structured as phase functions
# driven by main() at the bottom. See .claude/rules/platform.md for the full lifecycle.

# ── Paths & tunables ────────────────────────────────────────────────────────
WORKSPACE=/workspace
LOG_DIR="$WORKSPACE/logs"
REPO_DIR="$WORKSPACE/gpt-golf"
VENV=/opt/.venv
BAKED_EVENTD=/usr/local/bin/pod_eventd.py
HF_CACHE="$WORKSPACE/.cache/huggingface"
DEBUG_WAIT_SECS=30      # hold a failed pod open this long for SSH debugging before stopping
ACK_WAIT_SECS=120      # max wait for GPT Caddy to pull the final snapshot on a clean exit

POD_ID="${RUNPOD_POD_ID:-$(hostname)}"
LOG="$LOG_DIR/${POD_ID}.log"
# Exported for pod_eventd.py (events/log serving) and train_gpt.py. Snapshots live outside the
# repo so the git clone and `git reset --hard` below never trip over them.
export EVENTS_FILE="$LOG_DIR/${POD_ID}.events.jsonl" LOG_FILE="$LOG" SNAPSHOT_DIR="$WORKSPACE/snapshots"

# emit TYPE [KEY VALUE]... — append one JSON event line; never fatal to training
emit() { python3 - "$@" >> "$EVENTS_FILE" 2>/dev/null <<'PY' || true
import json, sys, time
NUM = {"exit_code", "gpu_count"}  # only known-numeric keys: blanket coercion would eat all-digit commit hashes
a = sys.argv[1:]; d = {"t": a[0], "ts": round(time.time(), 3)}
d.update({k: int(x) if k in NUM else x for k, x in zip(a[1::2], a[2::2])})
print(json.dumps(d))
PY
}

# require_env NAME... — fail fast (and auto-stop, since the EXIT trap is already armed) with a
# clear message instead of a cryptic set -u abort deep in code sync.
require_env() {
    local missing=() v
    for v in "$@"; do [ -n "${!v:-}" ] || missing+=("$v"); done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "==> FATAL: required env vars unset: ${missing[*]}" >&2
        emit phase phase "Boot failed (missing env: ${missing[*]})"
        exit 1
    fi
}

# ── Phases ──────────────────────────────────────────────────────────────────
setup_logging() {
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$LOG") 2>&1     # tee all subsequent output to a pod-named log
    echo "==> Logging to $LOG"
}

# The baked daemon serves boot events; swap_daemon upgrades it to the repo copy after code sync.
start_eventd() {
    mkdir -p "$SNAPSHOT_DIR"
    python3 "$BAKED_EVENTD" >> "$LOG_DIR/${POD_ID}.eventd.log" 2>&1 &
    EVENTD_PID=$!
}

setup_ssh() {
    mkdir -p /root/.ssh; chmod 700 /root/.ssh
    if [ -n "${PUBLIC_KEY:-}" ]; then
        echo "==> Injecting Public Key from environment..."
        printf "%s\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys
    fi
    # RunPod base image deletes host keys; regenerate any that are missing.
    for algo in rsa ecdsa ed25519; do
        [ -f "/etc/ssh/ssh_host_${algo}_key" ] || ssh-keygen -t "$algo" -f "/etc/ssh/ssh_host_${algo}_key" -q -N ''
    done
    mkdir -p /run/sshd; /usr/sbin/sshd
}

cleanup() {
    RET=$?
    echo "==> Training finished (exit $RET)."
    emit phase phase "Training finished" exit_code "$RET"

    if [ "$RET" -ne 0 ]; then
        echo "==> Non-zero exit, waiting ${DEBUG_WAIT_SECS}s before stopping pod..."
        sleep "$DEBUG_WAIT_SECS"
    else
        # Hold the pod until GPT Caddy acks the final snapshot (daemon writes the ack on download).
        # Successful runs otherwise stop within seconds, losing the last checkpoint.
        local ack_file="${SNAPSHOT_ACK_FILE:-/workspace/SNAPSHOT_ACK}"
        echo "==> Waiting up to ${ACK_WAIT_SECS}s for GPT Caddy to pull the final snapshot..."
        for _ in $(seq 1 "$ACK_WAIT_SECS"); do
            [ -f "$ack_file" ] && { echo "==> Final snapshot acked; stopping."; break; }
            sleep 1
        done
    fi

    if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "==> Sending GraphQL podStop mutation for pod ${RUNPOD_POD_ID}..."
        curl -fsS https://api.runpod.io/graphql \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -d "{\"query\":\"mutation { podStop(input: { podId: \\\"${RUNPOD_POD_ID}\\\" }) { id desiredStatus } }\"}" || echo "Warning: GraphQL Stop failed."
    else
        # Without credentials the wedge below never frees the pod — surface it so GPT Caddy can flag
        # the stranded (billing) pod instead of it silently hanging.
        echo "==> WARNING: RUNPOD_API_KEY/RUNPOD_POD_ID unset — cannot auto-stop; pod will wedge (manual stop required)."
        emit phase phase "Auto-stop skipped (no RunPod credentials)" exit_code "$RET"
    fi

    echo "==> Wedging container (sleep infinity) to prevent RunPod from restarting it."
    sleep infinity
}

sync_code() {
    emit phase phase "Syncing Code"
    mkdir -p "$WORKSPACE" && cd "$WORKSPACE"
    # Auth via env-injected git config (the actions/checkout pattern) so the token never lands in
    # the clone URL, .git/config, the process args, or the teed log that pod_eventd serves.
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        export GIT_CONFIG_COUNT=1
        export GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader"
        export GIT_CONFIG_VALUE_0="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)"
    fi
    local url="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
    if [ ! -d "gpt-golf/.git" ]; then
        echo "==> Fresh Clone..."
        git clone --branch "$BRANCH" --single-branch "$url" gpt-golf
    else
        echo "==> Updating Code..."
        cd gpt-golf && git fetch origin "$BRANCH" && git reset --hard "origin/$BRANCH"
    fi
    cd "$REPO_DIR"
    echo "==> Commit: $(git rev-parse --short HEAD) ($(git log -1 --format='%s'))"
    emit commit commit_hash "$(git rev-parse --short HEAD)" commit_msg "$(git log -1 --format='%s')"
}

# Hand off to the repo's event daemon if it changed since the image was built. Seq stays stable
# across the restart (line index of the persistent events file); GPT Caddy rides out the blip via
# its backoff/retry. Daemon changes therefore ship via git push, no image rebuild.
swap_daemon() {
    local repo_eventd="$REPO_DIR/scripts/pod_eventd.py"
    if [ -f "$repo_eventd" ] && ! cmp -s "$repo_eventd" "$BAKED_EVENTD"; then
        echo "==> Upgrading event daemon from repo..."
        kill "$EVENTD_PID" 2>/dev/null || true
        wait "$EVENTD_PID" 2>/dev/null || true
        python3 "$repo_eventd" >> "$LOG_DIR/${POD_ID}.eventd.log" 2>&1 &
        EVENTD_PID=$!
    fi
}

# venv baked into the image at /opt/.venv (local overlay, fast I/O); uv sync is a no-op safety check.
sync_deps() {
    source "$VENV/bin/activate"
    echo "==> Syncing Dependencies..."
    emit phase phase "Syncing Dependencies"
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync --no-dev --no-install-project
}

# DATA_VARIANT drives the dataset dir + config defaults; explicit DATA_PATH overrides. Two phases:
# a small blocking download (tokenizer + val + initial shards) gets training to torch.compile fast,
# then the full shard set streams in the background (--no-bundle so each shard lands atomically).
# train_gpt.py's data loader waits per-shard, pausing if it ever catches up.
prep_data() {
    local data_dir="${DATA_PATH:-data/datasets/fineweb10B_${DATA_VARIANT:-sp1024}}"
    local initial="${PREP_SHARDS_INITIAL:-$(( ${NPROC:-1} * 2 ))}"   # cover every dataloader worker (NPROC * num_workers=2)
    [ "$initial" -gt "${PREP_SHARDS:-10}" ] && initial="${PREP_SHARDS:-10}"   # never block on more than the streaming target
    if [ ! -d "$data_dir" ]; then
        # Export only on the download path: when data is already present (re-exec in a live pod)
        # these stay unset and train_gpt.py runs in static mode (globs the on-disk set, no waiting).
        export DATA_READY_SENTINEL="$data_dir/.download_complete" TRAIN_SHARDS="${PREP_SHARDS:-10}"
        rm -f "$DATA_READY_SENTINEL" "$DATA_READY_SENTINEL.tmp"   # never read a prior run's status on re-exec
        echo "==> Downloading initial data (repo=${DATA_REPO_ID:-sysrekt/parameter-golf} version=${DATA_VERSION:-main} variant=${DATA_VARIANT:-sp1024} initial=${initial} full=${PREP_SHARDS:-10} dir=${data_dir})..."
        emit phase phase "Downloading Data"
        HF_HUB_CACHE="$HF_CACHE" python3 data/cached_challenge_fineweb.py \
            --variant "${DATA_VARIANT:-sp1024}" --train-shards "$initial" --no-bundle
        echo "==> Streaming remaining shards in background (full=${PREP_SHARDS:-10})..."
        emit phase phase "Downloading Data (background)"
        (
            # if-guard so set -e can't abort before we record the exit code; the data loader reads it on shard miss
            if HF_HUB_CACHE="$HF_CACHE" python3 data/cached_challenge_fineweb.py \
                --variant "${DATA_VARIANT:-sp1024}" --train-shards "${PREP_SHARDS:-10}" --no-bundle; then rc=0; else rc=$?; fi
            echo "$rc" > "$DATA_READY_SENTINEL.tmp" && mv "$DATA_READY_SENTINEL.tmp" "$DATA_READY_SENTINEL"
        ) &
    fi
}

# Probe the Redis Inductor remote cache; sources the exported TORCHINDUCTOR_* vars on success.
configure_cache() {
    "$REPO_DIR/scripts/discover_cache.sh" || true
    [ -f /tmp/inductor_cache_env ] && source /tmp/inductor_cache_env || true
}

launch_training() {
    echo "==> GPU Info:"
    nvidia-smi || echo "warning: nvidia-smi not available"
    local gpu_name driver_ver cuda_ver gpu_count
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    # Driver version is a queryable field; CUDA version (max supported by the driver) only appears
    # in the header, so scrape it from there.
    driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1 || true)"
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    [ -n "$gpu_name" ] && emit gpu gpu_type "$gpu_name" gpu_count "$gpu_count" driver_version "$driver_ver" cuda_version "$cuda_ver" || true

    local train_script="${TRAIN_SCRIPT:-train_gpt.py}" nproc="${NPROC:-1}"
    echo "==> Starting Training ($train_script nproc=$nproc)..."
    emit phase phase "Starting Training"
    if [ "$nproc" -gt 1 ]; then
        # The base image ships a 16 KB libnccl.so.2 stub (single-GPU pulls skip ~206 MB); restore
        # the real wheel before torchrun so ProcessGroupNCCL has working collectives. The version is
        # read from the surviving dist-info — single source of truth with Dockerfile.base, so a base
        # bump self-follows. Fail-loud (set -e): training on the stub would silently corrupt collectives.
        echo "==> Multi-GPU run: restoring real NCCL..."
        local nccl_ver
        nccl_ver="$("$VENV/bin/python" -c 'from importlib.metadata import version; print(version("nvidia-nccl-cu13"))' 2>/dev/null || true)"
        [ -n "$nccl_ver" ] || { echo "==> FATAL: cannot resolve installed nvidia-nccl-cu13 version for NCCL restore" >&2; exit 1; }
        uv pip install --no-deps --reinstall-package nvidia-nccl-cu13 \
            --python "$VENV/bin/python" --index-url "$TORCH_INDEX" "nvidia-nccl-cu13==$nccl_ver"
        torchrun --standalone --nproc_per_node="$nproc" "$train_script"
    else
        python3 -u "$train_script"
    fi
}

main() {
    setup_logging
    start_eventd
    emit phase phase Boot
    setup_ssh
    rm -f "$WORKSPACE/STOP_LOOP"     # clear stale lock from shared volume (legacy)
    trap cleanup EXIT
    require_env BRANCH REPO_OWNER REPO_NAME
    sync_code
    swap_daemon
    sync_deps
    prep_data
    configure_cache
    launch_training
}

main "$@"
