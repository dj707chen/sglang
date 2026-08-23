#!/usr/bin/env bash
# One-shot snapshot: which SGLang processes exist, what role each plays, and
# what the server reports about itself. For a live view use ./monitor.py.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

echo "=== processes ==="
# The api-server/TokenizerManager process keeps the plain python command line;
# the children rename themselves via setproctitle to sglang::<role>.
# No detokenizer process means --rust: the Rust api-server, tokenizer and
# detokenizer all run as threads inside the scheduler process.
if ! ps -Ao pid,ppid,%cpu,rss,command | awk -v OFS='  ' '
    /awk|grep/ { next }
    /sglang::|sglang\.launch_server/ { rows[n++] = $0; if ($0 ~ /detokenizer/) py = 1 }
    END {
        for (i = 0; i < n; i++) {
            split(rows[i], f, " ")
            line = rows[i]
            if (line ~ /sglang::scheduler/)        role = "scheduler"
            else if (line ~ /sglang::detokenizer/) role = "detokenizer"
            else if (line ~ /launch_server/)
                role = py ? "api-server + TokenizerManager" : "launcher (Rust api-server lives in scheduler)"
            else role = "?"
            printf "  %-7s ppid=%-7s cpu=%5s%%  rss=%6.0fMB  %s\n", f[1], f[2], f[3], f[4]/1024, role
        }
        exit !n
    }
'; then
    echo "  (none running)"
fi

echo
echo "=== endpoints ($STUDY_BASE_URL) ==="
if ! curl -fsS -m 3 "$STUDY_BASE_URL/health" >/dev/null 2>&1; then
    echo "  server not responding"
    exit 0
fi

"$PY" - "$STUDY_BASE_URL" <<'PY'
import json, sys, urllib.request

base = sys.argv[1]

def get(path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read())

info = get("/model_info")
print(f"  model            {info.get('model_path')}")
if info.get("model_type"):  # the Rust api-server returns a smaller payload
    print(f"  architecture     {info['model_type']} {info.get('architectures')}")
print(f"  generation       {info.get('is_generation')}")

# The same metric name appears once per label set (e.g. per served model), so
# collapse duplicates by summing.
WANTED = [
    "sglang:context_len",
    "sglang:max_total_num_tokens",
    "sglang:page_size",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:num_requests_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
]
try:
    metrics_text = urllib.request.urlopen(base + "/metrics", timeout=5).read().decode()
except urllib.error.HTTPError:
    # The Rust api-server exposes no Prometheus endpoint.
    print("  metrics          (not exposed — Rust api-server, or no --enable-metrics)")
    raise SystemExit(0)

totals = {}
for line in metrics_text.splitlines():
    if line.startswith("#") or not line:
        continue
    head, _, value = line.rpartition(" ")
    name = head.split("{", 1)[0].strip()
    if name in WANTED:
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            pass
for name in WANTED:
    if name in totals:
        print(f"  {name.split(':')[1]:<16} {totals[name]:g}")
PY
