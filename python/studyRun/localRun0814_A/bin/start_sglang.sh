#!/usr/bin/env bash
# Launch the SGLang server with the study-run flag set, wait until it is really
# serving, and record the pid.
#
# Usage: start_sglang.sh [extra sglang args...]
#   e.g. start_sglang.sh --quantization mlx_q4

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

[[ -x "$PY" ]]         || die "venv python not found at $PY"
[[ -d "$MODEL_PATH" ]] || die "model not found at $MODEL_PATH"

if sglang_port_busy; then
  die "port $SGLANG_PORT is already in use. Run stop_sglang.sh first."
fi

TS="$(date +%Y%m%d-%H%M%S)"
echo "$TS" > "$PID_DIR/current_ts"
LOG="$LOG_DIR/server-$TS.log"

log "model      : $MODEL_PATH"
log "listen     : $SGLANG_URL"
log "server log : $LOG"
log "request log: $REQ_LOG_FILE"

# --- Flag rationale (Phase 6; each verified against server_args.py) ----------
#   --disable-cuda-graph ............ no CUDA on Apple Metal
#   --max-total-tokens .............. cap the KV pool (see env.sh)
#   --enable-metrics ................ exposes Prometheus registry at /metrics,
#                                     consumed by both Grafana (Ph3) and the TUI (Ph4)
#   --log-requests --log-requests-level 1
#                                     per-request lifecycle WITHOUT prompt/output text
#                                     (level 2+ would dump partial input/output)
#   --log-requests-format json ...... structured, so the TUI parses instead of regexing
#   --log-requests-target stdout DIR  both human scrollback and a machine-readable file
#   --uvicorn-access-log-exclude-prefixes /metrics /health
#                                     kills scrape noise but KEEPS real request access
#                                     logs -- strictly better than --log-level-http warning
#   --decode-log-interval 20 ........ ~1 scheduler stats line/sec at laptop decode rates
#   --enable-request-time-stats-logging
#                                     per-stage timings, feeds the Ph4 latency panel
#   --reasoning-parser qwen3 ........ split <think> out of `content` so token
#                                     accounting distinguishes reasoning from answer
ARGS=(
  --model-path "$MODEL_PATH"
  --host "$SGLANG_HOST" --port "$SGLANG_PORT"
  --disable-cuda-graph
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --enable-metrics
  --log-level "$LOG_LEVEL"
  --log-requests
  --log-requests-level "$LOG_REQUESTS_LEVEL"
  --log-requests-format json
  --log-requests-target stdout "$REQ_LOG_DIR"
  --uvicorn-access-log-exclude-prefixes /metrics /health
  --decode-log-interval "$DECODE_LOG_INTERVAL"
  --enable-request-time-stats-logging
  --reasoning-parser "$REASONING_PARSER"
  --crash-dump-folder "$CRASH_DIR"
)

# Re-floored latency buckets (see env.sh for why the defaults don't work here).
# shellcheck disable=SC2206
ARGS+=( --bucket-time-to-first-token $BUCKET_TTFT )
# shellcheck disable=SC2206
ARGS+=( --bucket-e2e-request-latency $BUCKET_E2E )

cd "$REPO_ROOT"
SGLANG_USE_MLX=1 nohup "$PY" -m sglang.launch_server "${ARGS[@]}" "$@" \
  > "$LOG" 2>&1 &
echo $! > "$SGLANG_PIDFILE"
PID="$(cat "$SGLANG_PIDFILE")"
log "launched pid=$PID, waiting for readiness..."

# Readiness: /health returns 503 until the warmup generate completes (measured in
# Phase 1), so poll for a 200 rather than treating the first non-200 as failure.
DEADLINE=$((SECONDS + 180))
while (( SECONDS < DEADLINE )); do
  if ! kill -0 "$PID" 2>/dev/null; then
    warn "process exited during startup. Last 30 log lines:"
    tail -30 "$LOG" >&2
    die "startup failed"
  fi
  if curl -sf -m 2 "$SGLANG_URL/health" >/dev/null 2>&1; then
    log "READY in ${SECONDS}s"
    "$BIN_DIR/status.sh" || true
    exit 0
  fi
  sleep 1
done

warn "not ready after 180s. Last 30 log lines:"
tail -30 "$LOG" >&2
die "readiness timeout"
