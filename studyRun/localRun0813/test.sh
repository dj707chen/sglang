#!/usr/bin/env bash
# Exercise the running server through both front doors, so the log shows the
# full path: HTTP -> TokenizerManager -> Scheduler -> TpModelWorker ->
# ModelRunner -> DetokenizerManager -> HTTP.
#
#   ./test.sh            # native /generate, OpenAI /v1/chat/completions, streaming, 4-way concurrency
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

curl -fsS -m 3 "$STUDY_BASE_URL/health" >/dev/null || {
    echo "server not up at $STUDY_BASE_URL — run ./start.sh" >&2
    exit 1
}

hr() { printf '\n\033[1m--- %s ---\033[0m\n' "$1"; }

hr "native /generate"
curl -sS "$STUDY_BASE_URL/generate" -H 'Content-Type: application/json' -d '{
  "text": "The three primary colors are",
  "sampling_params": {"temperature": 0, "max_new_tokens": 32}
}' | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d["text"]); print("meta:", d["meta_info"])'

hr "OpenAI /v1/chat/completions"
curl -sS "$STUDY_BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$STUDY_SERVED_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"In one sentence: what is a KV cache?\"}],
  \"max_tokens\": 64,
  \"temperature\": 0
}" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print("usage:", d["usage"])'

hr "streaming /generate (SSE, first 6 chunks)"
# head closes the pipe early, which SIGPIPEs curl; that is expected here.
set +o pipefail
curl -sS -N "$STUDY_BASE_URL/generate" -H 'Content-Type: application/json' -d '{
  "text": "Count from one to five:",
  "sampling_params": {"temperature": 0, "max_new_tokens": 24},
  "stream": true
}' 2>/dev/null | grep -o '"text":"[^"]*"' | head -6
set -o pipefail

hr "4 concurrent requests (watch the scheduler batch them)"
for i in 1 2 3 4; do
    curl -sS "$STUDY_BASE_URL/generate" -H 'Content-Type: application/json' -d "{
      \"text\": \"Give me fact number $i about the ocean:\",
      \"sampling_params\": {\"temperature\": 0.7, \"max_new_tokens\": 48}
    }" -o "$RUN_DIR/concurrent_$i.json" &
done
wait
for i in 1 2 3 4; do
    "$PY" -c "
import json
d = json.load(open('$RUN_DIR/concurrent_$i.json'))
m = d['meta_info']
print(f\"[$i] {m['completion_tokens']:>3} tok  cached={m.get('cached_tokens', 0):>3}  {d['text'].strip()[:70]!r}\")
"
done

hr "server-reported state"
"$(dirname "${BASH_SOURCE[0]}")/status.sh"
