#!/usr/bin/env bash
# Start the local SGLang server (MLX/Metal backend) in the background.
#
#   ./start.sh              # start, wait for readiness, print the process tree
#   ./start.sh --fg         # run in the foreground (Ctrl-C to stop)
#   ./start.sh --rust       # embedded Rust api-server instead of the Python one
#   STUDY_QUANT=mlx_q4 ./start.sh
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

FOREGROUND=0
RUST=0
for arg in "$@"; do
    case "$arg" in
        --fg) FOREGROUND=1 ;;
        --rust) RUST=1 ;;
        *) echo "unknown option: $arg (want --fg and/or --rust)" >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PY" ]]; then
    echo "error: no interpreter at $PY — see NOTES.md 'Setup' for how it was built" >&2
    exit 1
fi
if [[ ! -d "$STUDY_MODEL_PATH" ]]; then
    echo "error: model dir not found: $STUDY_MODEL_PATH" >&2
    exit 1
fi
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PID_FILE")); use ./stop.sh first" >&2
    exit 1
fi

ARGS=(
    -m sglang.launch_server
    --model-path "$STUDY_MODEL_PATH"
    --served-model-name "$STUDY_SERVED_NAME"
    --host "$STUDY_HOST"
    --port "$STUDY_PORT"
    # Metal has no CUDA graphs; the capture path must be off.
    --disable-cuda-graph
    --tp-size 1
    --log-level "$STUDY_LOG_LEVEL"
    --decode-log-interval "$STUDY_DECODE_LOG_INTERVAL"
)
[[ -n "$STUDY_QUANT" ]] && ARGS+=(--quantization "$STUDY_QUANT")

if (( RUST )); then
    # The Rust api-server runs as threads inside the scheduler process and
    # replaces the Python HTTP server, TokenizerManager and DetokenizerManager.
    # It needs the sglang.srt.server._core PyO3 extension; see NOTES.md.
    if ! "$PY" -c "import sglang.srt.server._core" 2>/dev/null; then
        echo "error: sglang.srt.server._core is not built — see NOTES.md 'Rust api-server'" >&2
        exit 1
    fi
    export SGLANG_RUST_SERVER=1
    # No uvicorn and no Python metrics registry in this mode.
else
    ARGS+=(
        # Exposes GET /metrics, which monitor.py scrapes.
        --enable-metrics
        # monitor.py polls /metrics once a second; keep that out of the access
        # log so the log tail stays readable.
        --uvicorn-access-log-exclude-prefixes /metrics /health /model_info
    )
    [[ "$STUDY_LOG_REQUESTS" == "1" ]] && ARGS+=(
        --log-requests --log-requests-level "$STUDY_LOG_REQUESTS_LEVEL"
    )
fi

if (( FOREGROUND )); then
    echo "==> $PY ${ARGS[*]}"
    exec "$PY" "${ARGS[@]}"
fi

: > "$SERVER_LOG"
nohup "$PY" "${ARGS[@]}" >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "launched pid $SERVER_PID  (log: $SERVER_LOG)"
printf 'waiting for %s/health' "$STUDY_BASE_URL"

# Model load + MLX warmup on an M3 is a handful of seconds; allow 5 minutes for
# a cold start where the weights are not yet in the page cache.
for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "server exited during startup — last 30 log lines:" >&2
        tail -30 "$SERVER_LOG" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -fsS -m 2 "$STUDY_BASE_URL/health" >/dev/null 2>&1; then
        echo " ok"
        echo
        "$(dirname "${BASH_SOURCE[0]}")/status.sh"
        exit 0
    fi
    printf '.'
    sleep 1
done

echo
echo "timed out waiting for readiness; see $SERVER_LOG" >&2
exit 1
