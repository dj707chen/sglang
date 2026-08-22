#!/usr/bin/env bash
# Stop the SGLang server and its whole process tree.
#
# The trap this avoids: `kill $(cat sglang.pid)` reaps only the parent, leaving
# `sglang::scheduler` alive holding the model, the KV pool and the ZMQ sockets.
# The next start then fails on a port collision with no obvious cause. So: kill
# the tree, escalate to SIGKILL, and verify the port is actually released.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# Portable (bash 3.2) array fill -- `mapfile` is bash 4+ and /bin/bash on macOS
# is still 3.2, so this must not depend on which bash won the PATH race.
collect_pids() {
  PIDS=()
  while IFS= read -r _line; do
    [[ -n "$_line" ]] && PIDS+=("$_line")
  done < <(sglang_pids)
}

collect_pids

if (( ${#PIDS[@]} == 0 )); then
  log "no sglang processes running"
else
  log "stopping ${#PIDS[@]} process(es): ${PIDS[*]}"
  ps -o pid,rss,comm= -p "${PIDS[@]}" 2>/dev/null | sed 's/^/    /' || true

  # Graceful first: lets sglang close ZMQ sockets and unlink its ipc:// files.
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done

  DEADLINE=$((SECONDS + 15))
  while (( SECONDS < DEADLINE )); do
    collect_pids
    (( ${#PIDS[@]} == 0 )) && break
    sleep 1
  done

  collect_pids
  if (( ${#PIDS[@]} > 0 )); then
    warn "still alive after 15s SIGTERM, escalating to SIGKILL: ${PIDS[*]}"
    for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
    sleep 2
  fi
fi

# Verify, don't assume.
collect_pids
(( ${#PIDS[@]} > 0 )) && die "could not kill: ${PIDS[*]}"

if sglang_port_busy; then
  warn "port $SGLANG_PORT still bound (TIME_WAIT, or a non-sglang listener):"
  lsof -nP -iTCP:"$SGLANG_PORT" -sTCP:LISTEN | sed 's/^/    /' >&2
else
  log "port $SGLANG_PORT released"
fi

rm -f "$SGLANG_PIDFILE"
log "stopped cleanly"
