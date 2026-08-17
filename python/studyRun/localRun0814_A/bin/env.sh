#!/usr/bin/env bash
# Single source of truth for the localRun0814_A study run.
# Sourced by every other script in bin/. Not meant to be executed directly.

set -euo pipefail

# --- Paths -------------------------------------------------------------------
# RUN_DIR is this script's parent's parent, resolved absolutely, so the scripts
# work from any cwd.
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$BIN_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUN_DIR/../.." && pwd)"
STUDY_ROOT="$(cd "$RUN_DIR/.." && pwd)"

VENV="$STUDY_ROOT/venvs/mps-py312"
PY="$VENV/bin/python"
MODEL_PATH="$STUDY_ROOT/models/Qwen3-0.6B"

LOG_DIR="$RUN_DIR/logs"
REQ_LOG_DIR="$LOG_DIR/requests"     # --log-requests-target writes <hostname>_<rank>.log here
CRASH_DIR="$LOG_DIR/crash"
PCAP_DIR="$RUN_DIR/pcap"
PID_DIR="$RUN_DIR/run"
OBS_DIR="$RUN_DIR/obs"

mkdir -p "$LOG_DIR" "$REQ_LOG_DIR" "$CRASH_DIR" "$PCAP_DIR" "$PID_DIR"

SGLANG_PIDFILE="$PID_DIR/sglang.pid"

# The request log filename is derived by sglang as <dir>/<hostname>_<rank>.log
# (see python/sglang/srt/utils/log_utils.py:_create_log_target_file).
REQ_LOG_FILE="$REQ_LOG_DIR/$(hostname)_0.log"

# --- Ports -------------------------------------------------------------------
SGLANG_HOST="127.0.0.1"
SGLANG_PORT="30000"
PROM_PORT="9090"
GRAFANA_PORT="3000"
NODE_EXPORTER_PORT="9100"

SGLANG_URL="http://${SGLANG_HOST}:${SGLANG_PORT}"

# --- Model / runtime knobs ---------------------------------------------------
# Cap the KV pool. Phase 1 measured sglang auto-sizing it to 12.03 GB / 112664
# tokens -- ~89% of free memory for a 0.6B model. That starves Prometheus,
# Grafana and the TUI, which share this machine's *unified* memory. 32768 tokens
# is still ~10x more than these experiments need.
MAX_TOTAL_TOKENS="32768"

# --- Logging (Phase 6) -------------------------------------------------------
# Values below were verified against python/sglang/srt/server_args.py, not
# assumed. See localRunPlan.md section 6 for the reasoning behind each.
LOG_LEVEL="info"
LOG_REQUESTS_LEVEL="1"              # choices 0..3; 1 = metadata + sampling params, no prompt text
DECODE_LOG_INTERVAL="20"            # default is 40; 20 gives ~1 scheduler stats line/sec here
REASONING_PARSER="qwen3"            # split Qwen3 <think> blocks out of `content`

# --- Histogram buckets (Phase 3) ---------------------------------------------
# sglang's default TTFT and e2e buckets both start at 0.1s -- they are tuned for
# large models on datacenter GPUs. Measured here: TTFT ~0.026s, e2e ~1.5s. With
# the stock buckets every fast request piles into the first bucket and
# histogram_quantile() cannot resolve anything below 100ms, so p50 TTFT would be
# a flat, meaningless 0.1. These re-floor the low end.
#   (ITL is left at its default -- it already starts at 0.002s, and the measured
#    ~0.017s mean lands mid-range with good resolution.)
BUCKET_TTFT="0.005 0.01 0.02 0.03 0.05 0.08 0.1 0.15 0.25 0.4 0.6 1.0 1.5 2.5 5.0 10.0"
BUCKET_E2E="0.05 0.1 0.2 0.35 0.5 0.75 1.0 1.5 2.0 3.0 5.0 8.0 15.0 30.0 60.0 120.0"

# --- Helpers -----------------------------------------------------------------
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[1;31m[%s] ERROR\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Is the sglang HTTP port accepting connections?
sglang_port_busy() {
  lsof -nP -iTCP:"$SGLANG_PORT" -sTCP:LISTEN >/dev/null 2>&1
}

# Every live pid belonging to this sglang instance: the recorded parent, its
# children, and any stray `sglang::` process. Deduplicated.
#
# The stray sweep matters: the scheduler and detokenizer are *children*, so
# killing only the pidfile pid orphans them holding the model, the KV pool and
# the ZMQ sockets -- and the next start then fails on a port collision.
sglang_pids() {
  local pids=""
  if [[ -f "$SGLANG_PIDFILE" ]]; then
    local parent
    parent="$(cat "$SGLANG_PIDFILE" 2>/dev/null || true)"
    if [[ -n "$parent" ]] && kill -0 "$parent" 2>/dev/null; then
      pids="$parent $(pgrep -P "$parent" 2>/dev/null || true)"
    fi
  fi
  # Match the setproctitle names set in scheduler.py / detokenizer_manager.py.
  pids="$pids $(pgrep -f 'sglang::' 2>/dev/null || true)"
  # Match a launch_server parent that outlived its pidfile.
  pids="$pids $(pgrep -f 'sglang.launch_server' 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [[ -n "${pids// /}" ]] && printf '%s\n' $pids | sort -un || true
}
