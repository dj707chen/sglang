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
  "text": "Which nations are the standng members of UN?",
  "sampling_params": {"temperature": 0, "max_new_tokens": 32}
}' | jq # | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d["text"]); print("meta:", d["meta_info"])'
