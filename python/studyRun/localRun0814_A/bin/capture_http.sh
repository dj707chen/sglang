#!/usr/bin/env bash
# Packet-level capture of the SGLang HTTP API on loopback.
#
# REQUIRES sudo. On macOS /dev/bpf* is root-only and this account is not in
# the access_bpf group, so tcpdump cannot run unprivileged. This script will
# prompt for your password -- it does not try to work around that.
#
# If you would rather not use sudo, `bin/tap.sh` records the same HTTP traffic
# at the application layer with no privileges (it loses the TCP/IP layer but
# gains decoded bodies and per-chunk SSE timings).
#
# Usage: capture_http.sh [seconds]     (default: run until ctrl-c)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

DURATION="${1:-}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="$PCAP_DIR/sglang-$TS.pcap"

# Capture the whole observability loop, not just client traffic:
#   30000 = client <-> sglang, and Prometheus <-> sglang /metrics scrapes
#   9090  = Grafana <-> Prometheus
#   9100  = Prometheus <-> node_exporter
FILTER="tcp port $SGLANG_PORT or tcp port $PROM_PORT or tcp port $NODE_EXPORTER_PORT"

log "interface : lo0"
log "filter    : $FILTER"
log "output    : $OUT"
log "decode it with: $PY $RUN_DIR/net/decode_pcap.py $OUT"
warn "tcpdump needs root; you will be prompted for your password."

ARGS=(-i lo0 -s 0 -w "$OUT" -U)
[[ -n "$DURATION" ]] && ARGS+=(-G "$DURATION" -W 1)

sudo tcpdump "${ARGS[@]}" "$FILTER"

log "capture written: $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
log "decode: $PY $RUN_DIR/net/decode_pcap.py $OUT"
