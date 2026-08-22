# Runbook — driving and inspecting the study run

Companion to [localRunPlan.md](localRunPlan.md) (what was built and why) and
[components.md](components.md) (how SGLang works here).

---

## Everyday commands

```bash
cd studyIterations/localRun0814_A

./bin/restart_all.sh      # the big red button: stop everything, start it all, smoke test
./bin/status.sh           # what's up, what's down, where the logs are
./bin/tui.sh              # live terminal dashboard (q to quit)
./bin/smoke.sh            # prove the server serves

./bin/start_sglang.sh     # server only     (extra args pass through)
./bin/stop_sglang.sh      # kills the whole process tree, verifies the port is freed
./bin/start_obs.sh        # Prometheus + Grafana + node_exporter
./bin/stop_obs.sh
./bin/stop_all.sh         # everything, plus a stray sweep
```

Grafana: <http://127.0.0.1:3000> (anonymous, no login) · Prometheus: <http://127.0.0.1:9090>

Useful variants:

```bash
./bin/start_sglang.sh --max-running-requests 2   # makes the queue actually fill up
./bin/start_sglang.sh --quantization mlx_q4      # 4-bit on the fly
./bin/tui.sh --iterations 3 --interval 2         # render frames to stdout, no TTY needed
```

---

## There are five different "traces" here

| # | Trace | Where | Needs sudo |
|---|---|---|---|
| 1 | Per-request lifecycle (JSON) | `logs/requests/<host>_0.log` | no |
| 2 | Scheduler iteration log | `logs/server-<ts>.log` | no |
| 3 | Application-layer HTTP capture | `pcap/http-<ts>.{jsonl,raw}` | no |
| 4 | Packet capture (pcap) | `pcap/sglang-<ts>.pcap` | **yes** |
| 5 | Per-iteration ZMQ telemetry | `pcap/fpm.jsonl` | no |

### 1. Per-request lifecycle

One JSON object per event, no prompt text (that's `--log-requests-level 1`):

```bash
# last 5 real requests, health checks filtered
grep -v HEALTH_CHECK logs/requests/*_0.log | tail -5

# pair up received/finished and show durations
python3 - <<'EOF'
import json, re, glob, datetime
ev = {}
for line in open(sorted(glob.glob("logs/requests/*_0.log"))[-1]):
    m = re.match(r'^\[[^\]]+\]\s+(\{.*\})', line.strip())
    if not m: continue
    r = json.loads(m.group(1))
    if r["rid"].startswith("HEALTH_CHECK_"): continue
    t = datetime.datetime.fromisoformat(r["timestamp"]).timestamp()
    ev.setdefault(r["rid"], {})[r["event"]] = t
for rid, e in list(ev.items())[-10:]:
    if "request.received" in e and "request.finished" in e:
        print(f'{rid[:8]}  {e["request.finished"]-e["request.received"]:.3f}s')
EOF
```

### 2. Scheduler iteration log

```bash
LOG=$(ls -t logs/server-*.log | head -1)

grep "Prefill batch" "$LOG" | tail -5      # admission + prefill
grep "Decode batch"  "$LOG" | tail -5      # per-iteration decode
grep "ReqTimeStats"  "$LOG" | tail -5      # queue vs forward time, per request
grep -E "MLX|MlxAttentionKVPool" "$LOG"    # what the MLX backend did at startup
```

`ReqTimeStats` is the most useful single line — it separates queueing from compute:

```
ReqTimeStats(rid=2a33f8cb…, input_len=3, output_len=12):
    queue_duration=0.14ms, forward_duration=1670.96ms
```

### 3. Application-layer HTTP capture (no sudo)

```bash
./bin/tap.sh &                                        # listens on :30001
curl http://127.0.0.1:30001/generate -d '{"text":"hi","sampling_params":{"max_new_tokens":8}}'
```

Then read it — the useful part is the **per-chunk SSE timings**, which give you
inter-token latency measured on the wire rather than from the server's own histogram:

```bash
python3 - <<'EOF'
import json, glob
f = sorted(glob.glob("pcap/http-*.jsonl"))[-1]
for line in open(f):
    r = json.loads(line)
    print(f"{r['method']} {r['path']} -> {r['status']}  "
          f"ttfb={r['ttfb_s']}s total={r['total_s']}s sse={r['sse_events']}")
    if r["streaming"]:
        ts = [c["dt"] for c in r["chunk_timings"]]
        gaps = [round(b-a, 4) for a, b in zip(ts, ts[1:])]
        print("   inter-chunk gaps (= inter-token latency):", gaps[:10])
EOF
```

`pcap/http-<ts>.raw` has the same exchanges as raw bytes, both directions, if you want to
see the literal wire format.

### 4. Packet capture — **this is the one you asked about**

```bash
./bin/capture_http.sh          # prompts for sudo; ctrl-C to stop
```

It captures ports 30000, 9090 and 9100, so you get the whole observability loop:
client↔sglang, Prometheus↔sglang, Grafana↔Prometheus, Prometheus↔node_exporter.

**Decode it with `net/decode_pcap.py`** — no Wireshark needed:

```bash
P=../venvs/mps-py312/bin/python
PCAP=pcap/sglang-20260814-182809.pcap

# a) What's in here at all?
$P net/decode_pcap.py $PCAP --summary

# b) The real traffic, with bodies (drop Prometheus scrape noise)
$P net/decode_pcap.py $PCAP --exclude /metrics --exclude /health --bodies

# c) Just the streaming completions, SSE events split out
$P net/decode_pcap.py $PCAP --only "chat/completions" --bodies

# d) Just the scrape traffic, if that's what you care about
$P net/decode_pcap.py $PCAP --only /metrics --summary
```

Example of (a) on a real capture:

```
packets: 1660   tcp streams with payload: 22
  127.0.0.1:51618  -> :30000   145 req   806.3 KB resp   145x GET /metrics
  127.0.0.1:51679  -> :30000     1 req     0.9 KB resp   1x POST /generate
  127.0.0.1:51682  -> :30000     1 req     7.5 KB resp   1x POST /v1/chat/completions
=== all requests seen ===
    145x  GET /metrics
      2x  POST /v1/chat/completions
      1x  GET /health
      1x  POST /generate
```

> **Prometheus dominates the capture.** At a 1 s scrape interval `/metrics` is ~145 of 149
> requests and ~800 KB of the traffic — the file grows about **1 MB/minute even when the
> model is idle**. Always `--exclude /metrics` when looking for inference traffic, and don't
> leave a capture running for hours.

**Stopping a capture:** it runs under `sudo`, so it needs `sudo` to stop. `./bin/stop_all.sh`
will detect it and tell you, but deliberately will not escalate on your behalf:

```bash
sudo pkill -f 'tcpdump.*port 30000'
```

Wireshark also opens these files if you prefer a GUI (`brew install --cask wireshark`);
`net/decode_pcap.py` exists so you don't have to.

### 5. Per-iteration ZMQ telemetry

```bash
./bin/start_sglang.sh --enable-forward-pass-metrics \
                      --forward-pass-metrics-ipc-name ipc:///tmp/sglang-fpm
../venvs/mps-py312/bin/python net/zmq_fpm_tap.py --endpoint ipc:///tmp/sglang-fpm.0 --seconds 20
```

⚠️ **On this machine you will only get idle heartbeats.** The transport works, but
`_emit_forward_pass_metrics()` returns early when `wall_time == 0.0`, and `wall_time` comes
from `DeviceTimer`, which is built on `torch.cuda.Event` and is only wrapped around forwards
under `model_executor/` — never under `hardware_backend/mlx/`. The script prints this
explanation when it sees heartbeats only. It would carry real data on a CUDA box.

---

## What is *not* capturable here

Raw ZMQ frames between TokenizerManager ↔ Scheduler ↔ DetokenizerManager. Single-node
SGLang wires those over `ipc://` **unix domain sockets**, which carry no packets, so
`tcpdump` cannot see them. `--enable-dp-attention` would move ZMQ to TCP, but it resolves to
`False` for a dense model like Qwen3-0.6B (it supports Qwen 2/3 **MoE**). See
[localRunPlan.md](localRunPlan.md) §5, Tier B.

You can still *see the topology* — `./bin/tui.sh` shows the bound unix sockets per process,
and [components.md](components.md) §7 explains the bind/connect asymmetry.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `port 30000 is already in use` | orphaned scheduler → `./bin/stop_all.sh` |
| Grafana panels empty | is `/metrics` up? `./bin/status.sh`. If you enabled `SGLANG_RUST_SERVER`, it returns 404 — that path has no `/metrics`. |
| TUI shows `UNREACHABLE` | server down or still warming (`/health` is 503 during warmup) |
| TUI queue panel always 0 | correct — `max_running_requests` is 4096. Use `--max-running-requests 2`. |
| pcap growing fast | Prometheus scrapes; stop the capture with `sudo pkill -f tcpdump` |
| `_core` import errors | the Rust server ext was built for a different Python; rebuild with `VIRTUAL_ENV` set explicitly |
