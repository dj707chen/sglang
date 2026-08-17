#!/usr/bin/env bash
# The panic button: stop everything this study run may have started, and sweep
# for strays. Safe to run when nothing is up.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

"$BIN_DIR/stop_sglang.sh" || warn "stop_sglang.sh reported a problem, continuing"

if [[ -x "$BIN_DIR/stop_obs.sh" ]]; then
  "$BIN_DIR/stop_obs.sh" || warn "stop_obs.sh reported a problem, continuing"
fi

# Stray sweep: anything still carrying an sglang marker.
strays=""
while IFS= read -r p; do [[ -n "$p" ]] && strays="$strays $p"; done < <(sglang_pids)
if [[ -n "${strays// /}" ]]; then
  warn "stray sglang processes survived:$strays"
  # shellcheck disable=SC2086
  kill -KILL $strays 2>/dev/null || true
fi

# The non-root HTTP tap (bin/tap.sh) -- ours to stop.
if pgrep -f "http_tap.py" >/dev/null 2>&1; then
  pkill -f "http_tap.py" 2>/dev/null || true
  log "http tap stopped"
fi
if pgrep -f "zmq_fpm_tap.py" >/dev/null 2>&1; then
  pkill -f "zmq_fpm_tap.py" 2>/dev/null || true
  log "fpm tap stopped"
fi

# A tcpdump left running by capture_http.sh holds the pcap open and keeps
# growing it. It runs under sudo, so it needs sudo to stop -- we won't escalate
# on your behalf, just tell you.
if pgrep -f "tcpdump.*port $SGLANG_PORT" >/dev/null 2>&1; then
  warn "a tcpdump capture on port $SGLANG_PORT is still running; stop it with:"
  warn "    sudo pkill -f 'tcpdump.*port $SGLANG_PORT'"
fi

log "stop_all complete"
