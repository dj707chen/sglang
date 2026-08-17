#!/usr/bin/env bash
# Shared configuration for the local SGLang study run on Apple Silicon.
# Sourced by start.sh / stop.sh / status.sh / test.sh. Override anything by
# exporting it before calling those scripts, e.g.
#
#   STUDY_PORT=30001 ./start.sh
#
# The knobs below are deliberately STUDY_*, not SGL_*: sglang's environ.py
# rewrites every SGL_* variable in the environment to SGLANG_* at import time
# and warns about each one, so an SGL_-prefixed name here would bury the log
# in deprecation warnings.
#
# shellcheck disable=SC2155

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# studyRun/, the parent of this run directory: shared across runs, and where
# the downloaded model weights live.
STUDY_ROOT="$(cd "$STUDY_DIR/.." && pwd)"
REPO_DIR="$(cd "$STUDY_ROOT/.." && pwd)"

# --- Python environment -------------------------------------------------
# Dedicated venv: the repo's default .venv targets the CUDA pyproject, which
# is not installable on macOS. This one was built from python/pyproject_other
# .toml's [srt_mps] extra (mlx + mlx-lm + torch 2.11 CPU/MPS).
export VENV="${VENV:-$STUDY_DIR/.venv}"
export PY="$VENV/bin/python"

# --- Model --------------------------------------------------------------
# Local directory, not a hub id: this network blocks the Hugging Face CDN
# (us.aws.cdn.hf.co -> "Generative AI Block"), so weights came from ModelScope
# and are read off disk. See NOTES.md.
export STUDY_MODEL_PATH="${STUDY_MODEL_PATH:-$STUDY_ROOT/models/Qwen3-0.6B}"
export STUDY_SERVED_NAME="${STUDY_SERVED_NAME:-qwen3-0.6b}"

# Optional on-the-fly MLX quantization: "" (bf16), mlx_q4, or mlx_q8.
export STUDY_QUANT="${STUDY_QUANT:-}"

# --- Server -------------------------------------------------------------
export STUDY_HOST="${STUDY_HOST:-127.0.0.1}"
export STUDY_PORT="${STUDY_PORT:-30000}"
export STUDY_BASE_URL="http://${STUDY_HOST}:${STUDY_PORT}"

# --- Logging ------------------------------------------------------------
# info level plus a scheduler stats line every 10 decode steps: enough to watch
# batches form and drain without drowning in output.
export STUDY_LOG_LEVEL="${STUDY_LOG_LEVEL:-info}"
export STUDY_DECODE_LOG_INTERVAL="${STUDY_DECODE_LOG_INTERVAL:-10}"

# Per-request Receive:/Finish: lines from the TokenizerManager. Off by default:
# every level of --log-requests-level prints the whole GenerateReqInput
# dataclass (~70 fields), which buries the scheduler lines. Turn it on to watch
# the TokenizerManager boundary itself:
#     STUDY_LOG_REQUESTS=1 ./start.sh
# Level 0 skips prompt text and sampling params, 1 adds sampling params,
# 2 truncates input/output to 2048 chars, 3 logs everything.
export STUDY_LOG_REQUESTS="${STUDY_LOG_REQUESTS:-0}"
export STUDY_LOG_REQUESTS_LEVEL="${STUDY_LOG_REQUESTS_LEVEL:-0}"

# --- Runtime dirs -------------------------------------------------------
export LOG_DIR="$STUDY_DIR/logs"
export RUN_DIR="$STUDY_DIR/run"
export SERVER_LOG="$LOG_DIR/server.log"
export PID_FILE="$RUN_DIR/server.pid"

mkdir -p "$LOG_DIR" "$RUN_DIR"

# --- Backend env vars ---------------------------------------------------
# SGLANG_USE_MLX=1 selects the MLX (Metal) runtime. Without it SGLang falls
# back to torch.mps, which covers far fewer ops.
export SGLANG_USE_MLX="${SGLANG_USE_MLX:-1}"
# Weights are on disk; never let the blocked HF CDN stall startup.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Tokenizers forks inside the scheduler process; silence its warning.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
