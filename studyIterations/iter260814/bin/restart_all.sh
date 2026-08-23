#!/usr/bin/env bash
# Full round-trip: tear everything down, bring it back, prove it works.
# This is the "re-run the whole serving servers" button.
#
# Usage: restart_all.sh [extra sglang args...]

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

log "=== 1/4  stopping everything ==="
"$BIN_DIR/stop_all.sh"

if [[ -x "$BIN_DIR/start_obs.sh" ]]; then
  log "=== 2/4  starting observability stack ==="
  "$BIN_DIR/start_obs.sh" || warn "observability stack failed to start, continuing without it"
else
  log "=== 2/4  observability stack not set up yet (Phase 3), skipping ==="
fi

log "=== 3/4  starting sglang ==="
"$BIN_DIR/start_sglang.sh" "$@"

log "=== 4/4  smoke test ==="
"$BIN_DIR/smoke.sh"

log "restart complete -- $SGLANG_URL"
