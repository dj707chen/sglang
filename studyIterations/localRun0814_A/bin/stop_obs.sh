#!/usr/bin/env bash
# Stop Prometheus, Grafana and node_exporter started by start_obs.sh.
# Only touches processes whose pids we recorded -- it will not kill a
# brew-services Grafana you started for some other purpose.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

stop_one() { # name pidfile
  local name="$1" pidf="$2" pid
  if [[ ! -f "$pidf" ]]; then
    log "$name not running (no pidfile)"
    return 0
  fi
  pid="$(cat "$pidf" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    log "$name not running (stale pidfile)"
    rm -f "$pidf"
    return 0
  fi

  kill -TERM "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "$name (pid $pid) ignored SIGTERM, sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pidf"
  log "$name stopped"
}

stop_one "grafana"       "$PID_DIR/grafana.pid"
stop_one "prometheus"    "$PID_DIR/prometheus.pid"
stop_one "node_exporter" "$PID_DIR/node_exporter.pid"
