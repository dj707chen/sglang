#!/usr/bin/env bash
# One-shot text summary of the whole study-run stack.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

hr() { printf '%s\n' "------------------------------------------------------------"; }

hr
printf 'SGLANG PROCESSES\n'
hr
found=0
while IFS= read -r p; do
  [[ -z "$p" ]] && continue
  found=1
  ps -o pid=,ppid=,rss=,%cpu=,comm= -p "$p" 2>/dev/null \
    | awk '{ printf "  pid=%-7s ppid=%-7s rss=%-8.1fMB cpu=%-6s %s\n", $1, $2, $3/1024, $4, $5 }'
done < <(sglang_pids)
(( found == 0 )) && printf '  (none)\n'

hr
printf 'ENDPOINTS\n'
hr
probe() { # name url
  if curl -sf -m 2 "$2" >/dev/null 2>&1; then
    printf '  %-26s \033[32mUP\033[0m    %s\n' "$1" "$2"
  else
    printf '  %-26s \033[31mdown\033[0m  %s\n' "$1" "$2"
  fi
}
probe "sglang /health"   "$SGLANG_URL/health"
probe "sglang /metrics"  "$SGLANG_URL/metrics"
probe "prometheus"       "http://127.0.0.1:$PROM_PORT/-/healthy"
probe "grafana"          "http://127.0.0.1:$GRAFANA_PORT/api/health"

if curl -sf -m 2 "$SGLANG_URL/get_server_info" >/dev/null 2>&1; then
  hr
  printf 'SERVER INFO\n'
  hr
  curl -s -m 5 "$SGLANG_URL/get_server_info" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
for k in ("model_path", "device", "tp_size", "dp_size", "pp_size",
          "max_total_num_tokens", "disable_overlap_schedule", "version"):
    if k in d:
        print(f"  {k:24s} = {d[k]}")
' 2>/dev/null || printf '  (could not parse)\n'
fi

hr
printf 'DISK\n'
hr
for d in "$LOG_DIR" "$PCAP_DIR"; do
  if [[ -d "$d" ]]; then
    printf '  %-10s %s\n' "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
  fi
done
printf '  %-10s %s\n' "model" "$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)"

hr
printf 'LATEST LOGS\n'
hr
latest="$(ls -t "$LOG_DIR"/server-*.log 2>/dev/null | head -1 || true)"
[[ -n "$latest" ]] && printf '  server : %s\n' "$latest" || printf '  server : (none)\n'
[[ -f "$REQ_LOG_FILE" ]] \
  && printf '  requests: %s (%s lines)\n' "$REQ_LOG_FILE" "$(wc -l < "$REQ_LOG_FILE" | tr -d ' ')" \
  || printf '  requests: (none yet)\n'
hr
