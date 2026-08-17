#!/usr/bin/env python3
"""Live view of a locally running SGLang server: its processes and its metrics.

Two panes, refreshed in place:

* **Processes** — one row per OS process in the engine, labelled with the role
  it plays (api-server / scheduler / detokenizer). The child processes rename
  themselves with ``setproctitle`` to ``sglang::<role>``, which is what makes
  the mapping possible from the outside.
* **Runtime** — a scrape of ``GET /metrics`` (the server must have been started
  with ``--enable-metrics``, which ``start.sh`` does), plus the tail of the
  server log.

Usage:
    ./monitor.py                       # defaults to http://127.0.0.1:30000
    ./monitor.py --url http://... --interval 0.5 --log-lines 12
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

import psutil

# Matches every process the engine owns, whether or not it has renamed itself.
PROC_PATTERNS = ("sglang::", "sglang.launch_server")

ROLES = (
    ("sglang::scheduler", "scheduler"),
    ("sglang::detokenizer", "detokenizer"),
    ("launch_server", "api-server + TokenizerManager"),
)

# In --rust mode there is no detokenizer process and no Python HTTP server: the
# api-server, tokenizer and detokenizer are Rust threads in the scheduler.
RUST_LAUNCHER_ROLE = "launcher (Rust api-server in scheduler)"

# name -> (label, formatter). Gauges are point-in-time; *_total are counters.
GAUGES = [
    ("sglang:num_running_reqs", "running reqs", "{:.0f}"),
    ("sglang:num_queue_reqs", "queued reqs", "{:.0f}"),
    ("sglang:gen_throughput", "gen tok/s", "{:.1f}"),
    ("sglang:token_usage", "KV usage", "{:.1%}"),
    ("sglang:num_used_tokens", "KV tokens used", "{:.0f}"),
    ("sglang:max_total_num_tokens", "KV capacity (tok)", "{:.0f}"),
    ("sglang:cache_hit_rate", "prefix cache hit", "{:.1%}"),
    ("sglang:num_requests_total", "requests total", "{:.0f}"),
    ("sglang:prompt_tokens_total", "prompt tokens", "{:.0f}"),
    ("sglang:generation_tokens_total", "generated tokens", "{:.0f}"),
]

# Histograms shown as running averages (sum / count).
AVERAGES = [
    ("sglang:time_to_first_token_seconds", "avg TTFT", "{:.3f}s"),
    ("sglang:inter_token_latency_seconds", "avg ITL", "{:.4f}s"),
    ("sglang:e2e_request_latency_seconds", "avg e2e latency", "{:.3f}s"),
]

CLEAR = "\x1b[H\x1b[2J"
DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"


def scrape(url: str, timeout: float = 2.0) -> dict[str, float]:
    """Prometheus text format -> {metric_name: value}, labels collapsed by sum."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode()
    except (urllib.error.URLError, OSError, TimeoutError):
        return {}

    out: dict[str, float] = collections.defaultdict(float)
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        name = head.split("{", 1)[0].strip()
        try:
            out[name] += float(value)
        except ValueError:
            continue
    return dict(out)


def find_processes(cache: dict[int, psutil.Process]) -> list[tuple[str, psutil.Process]]:
    """Engine processes as (role, Process), api-server first then children."""
    found = []
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or []) or proc.info["name"] or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(p in cmd for p in PROC_PATTERNS):
            continue
        role = next((label for needle, label in ROLES if needle in cmd), "worker")
        # Reuse the object across refreshes so cpu_percent() has a baseline.
        proc = cache.setdefault(proc.info["pid"], proc)
        found.append((role, proc))
    if not any(role == "detokenizer" for role, _ in found):
        found = [
            (RUST_LAUNCHER_ROLE if role == "api-server + TokenizerManager" else role, p)
            for role, p in found
        ]
    # Parent first, then its children in spawn order.
    parents = {"api-server + TokenizerManager", RUST_LAUNCHER_ROLE}
    found.sort(key=lambda rp: (rp[0] not in parents, rp[1].pid))
    return found


def render(args, cache, started: float) -> str:
    width = shutil.get_terminal_size((100, 40)).columns
    lines = [
        f"{BOLD}SGLang local monitor{RESET}  {args.url}"
        f"   {time.strftime('%H:%M:%S')}"
        f"   {DIM}watching for {int(time.time() - started)}s, ^C to quit{RESET}",
        "",
        f"{BOLD}processes{RESET}",
        f"  {'ROLE':<30}{'PID':>8}{'PPID':>8}{'CPU%':>8}{'RSS':>10}{'THREADS':>9}",
    ]

    procs = find_processes(cache)
    if not procs:
        lines.append(f"  {DIM}(no sglang processes){RESET}")
    total_rss = 0
    for role, proc in procs:
        try:
            with proc.oneshot():
                cpu = proc.cpu_percent(None)
                rss = proc.memory_info().rss
                ppid = proc.ppid()
                nthreads = proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cache.pop(proc.pid, None)
            continue
        total_rss += rss
        lines.append(
            f"  {role:<30}{proc.pid:>8}{ppid:>8}{cpu:>7.1f}%"
            f"{rss / 2**20:>9.0f}M{nthreads:>9}"
        )
    if procs:
        lines.append(f"  {DIM}{'total':<30}{'':>16}{'':>8}{total_rss / 2**20:>9.0f}M{RESET}")

    vm = psutil.virtual_memory()
    lines += [
        "",
        f"{BOLD}host{RESET}  cpu {psutil.cpu_percent(None):.0f}%"
        f"   mem {vm.percent:.0f}% of {vm.total / 2**30:.0f}G"
        f"   load {os.getloadavg()[0]:.2f}",
        "",
        f"{BOLD}runtime metrics{RESET}",
    ]

    metrics = scrape(args.url.rstrip("/") + "/metrics")
    if not metrics:
        lines.append(f"  {DIM}(no /metrics — server down, or started without --enable-metrics){RESET}")
    else:
        for name, label, fmt in GAUGES:
            if name in metrics:
                lines.append(f"  {label:<22}{fmt.format(metrics[name]):>14}")
        for name, label, fmt in AVERAGES:
            total, count = metrics.get(name + "_sum"), metrics.get(name + "_count")
            if total and count:
                lines.append(f"  {label:<22}{fmt.format(total / count):>14}")

    if args.log_lines and os.path.exists(args.log):
        lines += ["", f"{BOLD}{args.log}{RESET}"]
        lines += [
            f"  {DIM}{line[:width - 4]}{RESET}"
            for line in tail(args.log, args.log_lines)
        ]

    return "\n".join(lines)


def tail(path: str, n: int) -> list[str]:
    """Last n lines, read from the end so a long-running log stays cheap."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        block = min(size, 8192 * max(1, n // 40 + 1))
        fh.seek(size - block)
        data = fh.read(block)
    return data.decode(errors="replace").splitlines()[-n:]


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--log", default=os.path.join(here, "logs", "server.log"))
    parser.add_argument("--log-lines", type=int, default=8)
    args = parser.parse_args()

    cache: dict[int, psutil.Process] = {}
    started = time.time()
    # Prime the CPU counters so the first frame is not all zeros.
    find_processes(cache)
    psutil.cpu_percent(None)

    try:
        while True:
            frame = render(args, cache, started)
            sys.stdout.write(CLEAR + frame + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
