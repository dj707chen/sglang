#!/usr/bin/env bash
# Single source of truth for the iter260814 study run.
# Sourced by every other script in bin/. Not meant to be executed directly.

set -euo pipefail

# --- Paths -------------------------------------------------------------------
# RUN_DIR is this script's parent's parent, resolved absolutely, so the scripts
# work from any cwd.
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$BIN_DIR/.." && pwd)"
STUDY_ROOT="$(cd "$RUN_DIR/.." && pwd)"
# This run dir sits at <repo>/studyIterations/<run>, so the checkout root is two
# levels up -- i.e. $STUDY_ROOT's parent.
REPO_ROOT="$(cd "$RUN_DIR/../.." && pwd)"

# The serving env used to live at $STUDY_ROOT/venvs/mps-py312, which was a
# hand-rolled venv outside the repo. It is gone. The repo now builds the same
# thing at <repo>/.venv via python/setup_env.sh -- Python 3.12, torch + MLX,
# sglang installed editable -- so point at that instead of maintaining a second
# copy. Override with SGLANG_STUDY_VENV to test against a different env.
VENV="${SGLANG_STUDY_VENV:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"

# Weights are NOT in the repo and not in the venv. Fetch them with fetch_model.sh.
#
# Source is ModelScope, not HuggingFace, and that is deliberate: Netskope blocks
# the HF weight CDN here under category "Generative AI" (huggingface.co itself
# is fine, so you get metadata and no weights). ModelScope serves the same Qwen
# weights and is not blocked. See fetch_model.sh for the full diagnosis.
#
# bf16 rather than the mlx-community 4-bit build for the same reason -- the 4-bit
# repo is HF-only. bf16 at 1.4 GB is fine for a 0.6B model on this machine.
MODEL_SOURCE="${SGLANG_STUDY_MODEL_SOURCE:-modelscope}"   # modelscope | hf
MODEL_REPO="${SGLANG_STUDY_MODEL_REPO:-Qwen/Qwen3-0.6B}"
MODEL_PATH="${SGLANG_STUDY_MODEL:-$STUDY_ROOT/models/${MODEL_REPO##*/}}"

# Merged certifi + corporate-root CA bundle. Netskope intercepts TLS to the
# HuggingFace CDN with the `ca.jackhenry.goskope.com` root; curl trusts it via
# the macOS keychain but Python's bundled certifi does not, so Python dies with
# CERTIFICATE_VERIFY_FAILED where curl succeeds. fetch_model.sh builds this on
# demand. Harmless when absent -- nothing else here needs it.
CORP_CA_BUNDLE="${CORP_CA_BUNDLE:-$HOME/.config/certs/corp-ca-bundle.pem}"

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
