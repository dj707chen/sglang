#!/usr/bin/env bash
# Prove the server actually serves: native /generate, OpenAI chat, and streaming.
# Exits non-zero if any of the three fails, so it is usable as a gate.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

curl -sf -m 3 "$SGLANG_URL/health" >/dev/null 2>&1 || die "server not healthy at $SGLANG_URL"

fail=0

log "1/3  POST /generate"
if out=$(curl -sf -m 90 "$SGLANG_URL/generate" \
      -H 'Content-Type: application/json' \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}'); then
  printf '     -> %s\n' "$(printf '%s' "$out" | "$PY" -c 'import json,sys; print(repr(json.load(sys.stdin)["text"])[:160])')"
else
  warn "     /generate FAILED"; fail=1
fi

log "2/3  POST /v1/chat/completions"
if out=$(curl -sf -m 90 "$SGLANG_URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_PATH\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"max_tokens\":32,\"temperature\":0}"); then
  printf '%s' "$out" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
c = d["choices"][0]["message"]
print("     -> content   :", repr(c.get("content"))[:120])
print("     -> reasoning :", repr(c.get("reasoning_content"))[:120])
print("     -> usage     :", d.get("usage"))
'
else
  warn "     /v1/chat/completions FAILED"; fail=1
fi

log "3/3  streaming SSE"
n=$(curl -sf -N -m 60 "$SGLANG_URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_PATH\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three\"}],\"max_tokens\":24,\"temperature\":0,\"stream\":true}" \
    | grep -c '^data: ' || true)
if (( n > 1 )); then
  printf '     -> %s SSE chunks\n' "$n"
else
  warn "     streaming FAILED (got $n chunks)"; fail=1
fi

if (( fail )); then die "smoke test FAILED"; fi
log "smoke test PASSED"
