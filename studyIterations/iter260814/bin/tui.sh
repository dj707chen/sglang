#!/usr/bin/env bash
# Run the live TUI against the running server. Read-only.
#
# Usage: tui.sh [--interval 1.0] [--once] [--iterations N]

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

curl -sf -m 3 "$SGLANG_URL/health" >/dev/null 2>&1 \
  || warn "server is not healthy at $SGLANG_URL -- the TUI will show it as UNREACHABLE"

exec "$PY" "$RUN_DIR/tui/sglang_tui.py" \
  --url "$SGLANG_URL" \
  --request-log "$REQ_LOG_FILE" \
  "$@"
