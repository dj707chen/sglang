#!/usr/bin/env bash
# Start Prometheus + Grafana + node_exporter for this study run.
#
# Deliberately NOT `brew services`: those install login-time launchd agents and
# read /opt/homebrew/etc configs. Everything here runs from this repo's config,
# writes to this run's data dirs, and dies when you say so.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROM_BIN="$(command -v prometheus || true)"
GRAF_BIN="$(command -v grafana || true)"
NODE_BIN="$(command -v node_exporter || true)"
[[ -n "$PROM_BIN" ]] || die "prometheus not found. brew install prometheus grafana node_exporter"
[[ -n "$GRAF_BIN" ]] || die "grafana not found. brew install prometheus grafana node_exporter"
[[ -n "$NODE_BIN" ]] || die "node_exporter not found. brew install prometheus grafana node_exporter"

GRAF_HOME="/opt/homebrew/opt/grafana/share/grafana"
[[ -d "$GRAF_HOME" ]] || die "grafana homepath not found at $GRAF_HOME"

DATA_DIR="$OBS_DIR/data"
PROM_DATA="$DATA_DIR/prometheus"
GRAF_DATA="$DATA_DIR/grafana"
GRAF_PROV="$DATA_DIR/grafana-provisioning"   # rendered from the tracked templates
GRAF_LOGS="$LOG_DIR/grafana"
mkdir -p "$PROM_DATA" "$GRAF_DATA" "$GRAF_PROV/datasources" "$GRAF_PROV/dashboards" "$GRAF_LOGS"

DASH_DIR="$OBS_DIR/grafana/dashboards"
HOME_DASH="$DASH_DIR/sglang-runtime.json"

start_one() { # name binary pidfile logfile args...
  local name="$1" bin="$2" pidf="$3" logf="$4"; shift 4
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
    log "$name already running (pid $(cat "$pidf"))"
    return 0
  fi
  nohup "$bin" "$@" > "$logf" 2>&1 &
  echo $! > "$pidf"
  log "$name started (pid $(cat "$pidf")) -> $logf"
}

# --- node_exporter -----------------------------------------------------------
start_one "node_exporter" "$NODE_BIN" "$PID_DIR/node_exporter.pid" "$LOG_DIR/node_exporter.log" \
  --web.listen-address="127.0.0.1:$NODE_EXPORTER_PORT"

# --- Prometheus --------------------------------------------------------------
start_one "prometheus" "$PROM_BIN" "$PID_DIR/prometheus.pid" "$LOG_DIR/prometheus.log" \
  --config.file="$OBS_DIR/prometheus.yml" \
  --storage.tsdb.path="$PROM_DATA" \
  --storage.tsdb.retention.time=7d \
  --web.listen-address="127.0.0.1:$PROM_PORT" \
  --web.enable-lifecycle

# --- Grafana -----------------------------------------------------------------
# Render the tracked provisioning templates with absolute paths. Grafana's
# provisioning files can't reference relative paths, but hardcoding an absolute
# path into a checked-in file would break on any other machine -- so the
# templates carry placeholders and we substitute at start time.
sed "s|__DASHBOARD_DIR__|$DASH_DIR|g" \
  "$OBS_DIR/grafana/provisioning/dashboards/dashboards.yml" \
  > "$GRAF_PROV/dashboards/dashboards.yml"
cp "$OBS_DIR/grafana/provisioning/datasources/prometheus.yml" \
  "$GRAF_PROV/datasources/prometheus.yml"

GRAF_INI="$DATA_DIR/grafana.ini"
sed "s|__HOME_DASHBOARD__|$HOME_DASH|g" "$OBS_DIR/grafana/grafana.ini" > "$GRAF_INI"

start_one "grafana" "$GRAF_BIN" "$PID_DIR/grafana.pid" "$LOG_DIR/grafana.log" \
  server \
  --homepath "$GRAF_HOME" \
  --config "$GRAF_INI" \
  cfg:default.paths.data="$GRAF_DATA" \
  cfg:default.paths.logs="$GRAF_LOGS" \
  cfg:default.paths.provisioning="$GRAF_PROV"

# --- Wait for readiness ------------------------------------------------------
wait_for() { # name url deadline_s
  local name="$1" url="$2" deadline=$((SECONDS + $3))
  while (( SECONDS < deadline )); do
    curl -sf -m 2 "$url" >/dev/null 2>&1 && { log "$name ready"; return 0; }
    sleep 1
  done
  warn "$name did not become ready at $url"
  return 1
}

rc=0
wait_for "node_exporter" "http://127.0.0.1:$NODE_EXPORTER_PORT/metrics" 20 || rc=1
wait_for "prometheus"    "http://127.0.0.1:$PROM_PORT/-/healthy"        30 || rc=1
wait_for "grafana"       "http://127.0.0.1:$GRAFANA_PORT/api/health"    60 || rc=1

if (( rc == 0 )); then
  log "Grafana   : http://127.0.0.1:$GRAFANA_PORT  (anonymous admin, no login needed)"
  log "Prometheus: http://127.0.0.1:$PROM_PORT"
else
  warn "one or more components failed; check $LOG_DIR/{prometheus,grafana,node_exporter}.log"
fi
exit $rc
