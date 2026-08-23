#!/usr/bin/env bash
# Run the non-root recording HTTP proxy (net/http_tap.py).
#
# Records every request/response to pcap/http-<ts>.{jsonl,raw}. Send clients to
# the tap port instead of the server port:
#
#     ./bin/tap.sh &
#     curl http://127.0.0.1:30001/generate -d '{...}'
#
# Ctrl-C to stop. Unlike bin/capture_http.sh this needs no privileges.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TAP_PORT="${TAP_PORT:-30001}"

curl -sf -m 3 "$SGLANG_URL/health" >/dev/null 2>&1 \
  || warn "sglang is not healthy at $SGLANG_URL; the tap will return 502s"

log "tap  : http://127.0.0.1:$TAP_PORT  ->  $SGLANG_URL"
log "out  : $PCAP_DIR/http-<ts>.{jsonl,raw}"

exec "$PY" "$RUN_DIR/net/http_tap.py" \
  --listen "$TAP_PORT" \
  --target "${SGLANG_HOST}:${SGLANG_PORT}" \
  --outdir "$PCAP_DIR" "$@"
