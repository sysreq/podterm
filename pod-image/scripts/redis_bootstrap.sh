#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/data"
mkdir -p "$DATA_DIR"

# Compile-cache oriented defaults (override via env vars as needed).
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_BIND="${REDIS_BIND:-0.0.0.0}"
REDIS_SAVE_SCHEDULE="${REDIS_SAVE_SCHEDULE:-}"
REDIS_MAXMEMORY_POLICY="${REDIS_MAXMEMORY_POLICY:-allkeys-lfu}"
REDIS_CLIENT_OUTPUT_BUFFER_LIMIT="${REDIS_CLIENT_OUTPUT_BUFFER_LIMIT:-normal 256mb 128mb 60}"
REDIS_PROTO_MAX_BULK_LEN="${REDIS_PROTO_MAX_BULK_LEN:-512mb}"

# Default maxmemory to 70% of system RAM (minimum 1 GiB) if not explicitly set.
if [ -n "${REDIS_MAXMEMORY:-}" ]; then
    REDIS_MAXMEMORY_VALUE="${REDIS_MAXMEMORY}"
elif [ -r /proc/meminfo ]; then
    MEM_KB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    if [[ "${MEM_KB}" =~ ^[0-9]+$ ]] && [ "${MEM_KB}" -gt 0 ]; then
        REDIS_MAXMEMORY_VALUE="$((MEM_KB * 1024 * 70 / 100))"
        [ "${REDIS_MAXMEMORY_VALUE}" -lt 1073741824 ] && REDIS_MAXMEMORY_VALUE=1073741824
    else
        REDIS_MAXMEMORY_VALUE="4gb"
    fi
else
    REDIS_MAXMEMORY_VALUE="4gb"
fi

# IO threads: use ~half the cores, capped to 8. Keep reads on threaded path only when >1.
CPU_COUNT="$(nproc 2>/dev/null || echo 2)"
if [[ "${CPU_COUNT}" =~ ^[0-9]+$ ]] && [ "${CPU_COUNT}" -gt 1 ]; then
    REDIS_IO_THREADS="$((CPU_COUNT / 2))"
    [ "${REDIS_IO_THREADS}" -gt 8 ] && REDIS_IO_THREADS=8
else
    REDIS_IO_THREADS=1
fi
[ "${REDIS_IO_THREADS}" -lt 1 ] && REDIS_IO_THREADS=1
REDIS_IO_THREADS_DO_READS="no"
[ "${REDIS_IO_THREADS}" -gt 1 ] && REDIS_IO_THREADS_DO_READS="yes"

# 1. SSH
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    printf "%s\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
for algo in rsa ecdsa ed25519; do
    [ -f "/etc/ssh/ssh_host_${algo}_key" ] || ssh-keygen -t "$algo" -f "/etc/ssh/ssh_host_${algo}_key" -q -N ''
done
/usr/sbin/sshd

# 2. Cleanup trap — in-memory cache, nothing to persist.
cleanup() {
    redis-cli SHUTDOWN NOSAVE 2>/dev/null || true
}
trap cleanup EXIT

# 3. Start Redis (background so trap can fire)
echo "==> Redis tuning: port=${REDIS_PORT} io_threads=${REDIS_IO_THREADS} maxmemory=${REDIS_MAXMEMORY_VALUE} policy=${REDIS_MAXMEMORY_POLICY}"
redis-server \
    --port "${REDIS_PORT}" \
    --bind "${REDIS_BIND}" \
    --dir "$DATA_DIR" \
    --dbfilename dump.rdb \
    --save "${REDIS_SAVE_SCHEDULE}" \
    --appendonly no \
    --rdbcompression no \
    --stop-writes-on-bgsave-error no \
    --maxmemory "${REDIS_MAXMEMORY_VALUE}" \
    --maxmemory-policy "${REDIS_MAXMEMORY_POLICY}" \
    --maxmemory-samples 10 \
    --lazyfree-lazy-eviction yes \
    --protected-mode no \
    --client-output-buffer-limit "${REDIS_CLIENT_OUTPUT_BUFFER_LIMIT}" \
    --proto-max-bulk-len "${REDIS_PROTO_MAX_BULK_LEN}" \
    --io-threads "${REDIS_IO_THREADS}" \
    --io-threads-do-reads "${REDIS_IO_THREADS_DO_READS}" \
    --daemonize no &

REDIS_PID=$!
echo "==> Redis ready on port ${REDIS_PORT} (PID: $REDIS_PID)"

wait $REDIS_PID
