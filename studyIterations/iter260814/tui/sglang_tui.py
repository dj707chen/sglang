#!/usr/bin/env python3
"""Live terminal dashboard for a running SGLang server.

Deliberately complements Grafana rather than duplicating it: Grafana answers
"what are the trends", this answers "what are the *processes* doing right now".
The process-tree and IPC-topology panels have no Grafana equivalent -- they are
the whole point, because making the multi-process architecture visible is what
this study run is about.

Dependencies: rich, psutil, prometheus_client -- all already in the venv. No new
installs, and notably no `textual`.

Usage:
    python3 sglang_tui.py [--url URL] [--interval SEC] [--once] [--iterations N]

    --once / --iterations   render N frames to stdout and exit (non-interactive;
                            used for testing and for pasting into notes)
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import psutil
from prometheus_client.parser import text_string_to_metric_families
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Palette mirrors the Grafana dashboard so the two read as one system.
C_BLUE, C_ORANGE, C_AQUA = "#3987e5", "#d95926", "#199e70"
C_YELLOW, C_MAGENTA, C_VIOLET = "#c98500", "#d55181", "#9085e9"
C_DIM = "grey42"

SPARK = "▁▂▃▄▅▆▇█"

# sglang tags every process it forks with setproctitle; these are the exact
# prefixes used in scheduler.py / detokenizer_manager.py / data_parallel_controller.py.
SGLANG_PROC_MARKERS = ("sglang::",)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One /metrics scrape, flattened for lookup."""

    when: float
    gauges: Dict[str, float] = field(default_factory=dict)
    # name -> {labelkey: value}, for label-split counters
    by_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # histogram name -> sorted [(le, cumulative_count)]
    hists: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    hist_sum: Dict[str, float] = field(default_factory=dict)
    hist_count: Dict[str, float] = field(default_factory=dict)


def scrape(url: str, timeout: float = 2.0) -> Optional[Sample]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            body = fh.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None

    s = Sample(when=time.time())
    for fam in text_string_to_metric_families(body):
        for m in fam.samples:
            name, labels, val = m.name, m.labels, m.value
            if name.endswith("_bucket"):
                base = name[: -len("_bucket")]
                le = labels.get("le")
                if le is None:
                    continue
                le_f = float("inf") if le == "+Inf" else float(le)
                s.hists.setdefault(base, []).append((le_f, val))
            elif name.endswith("_sum"):
                s.hist_sum[name[: -len("_sum")]] = val
            elif name.endswith("_count"):
                s.hist_count[name[: -len("_count")]] = val
            else:
                # Gauges/counters. Sum across label sets for the plain view, and
                # keep a per-"mode"/"stage" split where the label exists.
                s.gauges[name] = s.gauges.get(name, 0.0) + val
                for key in ("mode", "stage"):
                    if key in labels:
                        s.by_label.setdefault(name, {})
                        k = labels[key]
                        s.by_label[name][k] = s.by_label[name].get(k, 0.0) + val
    for k in s.hists:
        s.hists[k].sort()
    return s


def hist_quantile(sample: Sample, base: str, q: float) -> Optional[float]:
    """Linear-interpolated quantile from cumulative histogram buckets.

    Same approach as Prometheus histogram_quantile: find the bucket containing
    the target rank and interpolate within it. Returns None if no observations.
    """
    buckets = sample.hists.get(base)
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= target:
            if le == float("inf"):
                return prev_le
            if c == prev_c:
                return le
            frac = (target - prev_c) / (c - prev_c)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_c = le, c
    return None


def rate(
    cur: Sample, prev: Optional[Sample], name: str, label: Optional[str] = None
) -> Optional[float]:
    if prev is None:
        return None
    dt = cur.when - prev.when
    if dt <= 0:
        return None
    if label is None:
        a = cur.gauges.get(name)
        b = prev.gauges.get(name)
    else:
        a = cur.by_label.get(name, {}).get(label)
        b = prev.by_label.get(name, {}).get(label)
    if a is None or b is None:
        return None
    d = a - b
    if d < 0:  # counter reset (server restart)
        return None
    return d / dt


# --------------------------------------------------------------------------- #
# Processes
# --------------------------------------------------------------------------- #
@dataclass
class ProcInfo:
    pid: int
    ppid: int
    name: str
    role: str
    rss: float
    cpu: float
    threads: int
    uptime: float


def find_procs() -> List[ProcInfo]:
    """Locate the sglang process family.

    Anchors on the `sglang::` setproctitle names and on a `sglang.launch_server`
    cmdline, then adds the common parent so the tree is complete.
    """
    hits: Dict[int, psutil.Process] = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            nm = p.info["name"] or ""
            cmd = " ".join(p.info["cmdline"] or [])
            # IMPORTANT: on macOS psutil.name() returns "Python" for the forked
            # workers -- setproctitle rewrites argv, which surfaces in cmdline,
            # not in name. Checking name alone silently finds only the parent.
            hay = nm + " " + cmd
            if (
                any(m in hay for m in SGLANG_PROC_MARKERS)
                or "sglang.launch_server" in cmd
            ):
                hits[p.info["pid"]] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Pull in parents, then their other children.
    #
    # These must be two separate passes. Folding the sibling sweep inside an
    # `if parent not in hits` guard means that when the parent was ALREADY
    # matched directly (it always is -- its cmdline has sglang.launch_server),
    # the sweep never runs and multiprocessing's resource_tracker goes missing.
    # Showing it, explicitly labelled as not-an-sglang-component, is the point:
    # it stops the component count being read as 4.
    for pid in list(hits):
        try:
            par = psutil.Process(pid).parent()
            if par and par.pid not in hits and "sglang" in " ".join(par.cmdline()):
                hits[par.pid] = par
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for pid in list(hits):
        try:
            for ch in psutil.Process(pid).children():
                hits.setdefault(ch.pid, ch)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    out: List[ProcInfo] = []
    now = time.time()
    for pid, p in hits.items():
        try:
            with p.oneshot():
                nm = p.name() or ""
                cmd = " ".join(p.cmdline())
                out.append(
                    ProcInfo(
                        pid=pid,
                        ppid=p.ppid(),
                        name=nm,
                        role=classify(nm + " " + cmd, cmd),
                        rss=p.memory_info().rss / 1024 / 1024,
                        cpu=p.cpu_percent(interval=None),
                        threads=p.num_threads(),
                        uptime=now - p.create_time(),
                    )
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    out.sort(key=lambda x: (x.ppid, x.pid))
    return out


def classify(name: str, cmdline: str) -> str:
    """Map a process to the sglang component it implements.

    The resource_tracker distinction matters: it is stdlib multiprocessing
    bookkeeping, NOT an sglang component, and calling it one would misrepresent
    the architecture.
    """
    if "sglang::scheduler" in name:
        return "Scheduler +TpWorker +ModelRunner"
    if "sglang::detokenizer_router" in name:
        return "DetokenizerRouter"
    if "sglang::detokenizer" in name:
        return "DetokenizerManager"
    if "sglang::data_parallel_controller" in name:
        return "DataParallelController"
    if "sglang::tokenizer_worker" in name:
        return "TokenizerWorker"
    if "resource_tracker" in cmdline:
        return "resource_tracker (stdlib)"
    if "sglang.launch_server" in cmdline:
        return "HTTP server + TokenizerManager"
    return "?"


ROLE_STYLE = {
    "Scheduler +TpWorker +ModelRunner": C_ORANGE,
    "DetokenizerManager": C_AQUA,
    "DetokenizerRouter": C_AQUA,
    "DataParallelController": C_VIOLET,
    "TokenizerWorker": C_BLUE,
    "HTTP server + TokenizerManager": C_BLUE,
    "resource_tracker (stdlib)": C_DIM,
}


def ipc_sockets(pids: Sequence[int]) -> Dict[int, List[str]]:
    """ZMQ unix-domain socket paths held per process, via lsof.

    This is how the tokenizer <-> scheduler <-> detokenizer wiring is made
    visible: single-node sglang uses ipc:// unix sockets (server_args.py),
    not TCP, so they never appear in netstat.
    """
    if not pids:
        return {}
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-a", "-p", ",".join(str(p) for p in pids), "-U"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    res: Dict[int, List[str]] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        path = parts[-1]
        if path.startswith("/") and ("/T/" in path or "tmp" in path):
            res.setdefault(pid, [])
            if path not in res[pid]:
                res[pid].append(path)
    return res


# --------------------------------------------------------------------------- #
# Request log tail
# --------------------------------------------------------------------------- #
LOGLINE = re.compile(r"^\[[^\]]+\]\s+(\{.*\})\s*$")


def _parse_ts(s: str) -> Optional[float]:
    """ISO-8601 timestamp from the request log -> epoch seconds."""
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


class RequestTail:
    """Tail the JSON request log written by --log-requests-target.

    Health-check requests are filtered: every /health probe emits a real
    request pair with rid prefixed HEALTH_CHECK_, and at a 1s poll they would
    crowd out every genuine request.
    """

    def __init__(self, path: str, keep: int = 60):
        self.path = path
        self.pos = 0
        self.events: collections.deque = collections.deque(maxlen=keep)
        self.started: Dict[str, float] = {}
        self.skipped_health = 0
        if os.path.exists(path):
            self.pos = os.path.getsize(path)  # start at the tail, not the top

    def poll(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            size = os.path.getsize(self.path)
            if size < self.pos:  # rotated (TimedRotatingFileHandler, hourly)
                self.pos = 0
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
                self.pos = fh.tell()
        except OSError:
            return

        for line in chunk.splitlines():
            m = LOGLINE.match(line.strip())
            if not m:
                continue
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            rid = rec.get("rid", "")
            if rid.startswith("HEALTH_CHECK_"):
                self.skipped_health += 1
                continue
            ev = rec.get("event", "")
            # Duration must come from the log's own timestamps, not from
            # wall-clock at parse time: a single poll usually reads BOTH the
            # received and finished lines for a request, so `now - now` scored
            # every request as 0.00s.
            ts = _parse_ts(rec.get("timestamp", ""))
            if ev == "request.received":
                if ts is not None:
                    self.started[rid] = ts
            elif ev == "request.finished":
                start = self.started.pop(rid, None)
                dur = (ts - start) if (ts is not None and start is not None) else None
                obj = rec.get("obj") or {}
                sp = obj.get("sampling_params") or {}
                self.events.appendleft(
                    {
                        "rid": rid[:8],
                        "dur": dur,
                        "max_new": sp.get("max_new_tokens"),
                        "temp": sp.get("temperature"),
                        "t": rec.get("timestamp", "")[11:19],
                    }
                )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def spark(series: Sequence[Optional[float]], width: int = 28) -> str:
    vals = [v for v in list(series)[-width:] if v is not None]
    if not vals:
        return "[dim]no data[/dim]"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def fmt_secs(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v < 1e-3:
        return f"{v * 1e6:.0f}µs"
    if v < 1:
        return f"{v * 1e3:.1f}ms"
    return f"{v:.2f}s"


def fmt_dur(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def panel_processes(procs: List[ProcInfo]) -> Panel:
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column("pid", justify="right", style=C_DIM, no_wrap=True, width=6)
    # no_wrap + ellipsis: a long role must degrade to "Schedul…" rather than
    # wrap onto three lines and blow the panel height out.
    t.add_column("role", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("rss", justify="right", no_wrap=True, width=7)
    t.add_column("cpu", justify="right", no_wrap=True, width=5)
    t.add_column("thr", justify="right", style=C_DIM, no_wrap=True, width=4)
    t.add_column("up", justify="right", style=C_DIM, no_wrap=True, width=7)
    t.add_row(
        Text("PID", style="bold"),
        Text("COMPONENT", style="bold"),
        Text("RSS", style="bold"),
        Text("CPU", style="bold"),
        Text("THR", style="bold"),
        Text("UP", style="bold"),
    )
    if not procs:
        t.add_row("—", Text("no sglang processes found", style="red"), "", "", "", "")
    for p in procs:
        style = ROLE_STYLE.get(p.role, "white")
        prefix = "└─ " if p.ppid != 1 and any(q.pid == p.ppid for q in procs) else ""
        t.add_row(
            str(p.pid),
            Text(prefix + p.role, style=style),
            f"{p.rss:,.0f}M",
            f"{p.cpu:.0f}%",
            str(p.threads),
            fmt_dur(p.uptime),
        )
    total = sum(p.rss for p in procs)
    t.add_row(
        "",
        Text("total", style="bold"),
        Text(f"{total:,.0f}M", style="bold"),
        "",
        "",
        "",
    )
    return Panel(t, title="[bold]Processes[/bold]", border_style=C_DIM)


def panel_scheduler(cur: Optional[Sample], prev: Optional[Sample]) -> Panel:
    # Explicit ratios on BOTH columns: with only the value column constrained,
    # rich can collapse the label column to zero width in a narrow panel and the
    # labels silently vanish.
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(ratio=3, no_wrap=True, overflow="ellipsis")
    t.add_column(ratio=2, justify="right", no_wrap=True)
    if cur is None:
        return Panel(
            Text("server unreachable", style="red"),
            title="[bold]Scheduler[/bold]",
            border_style="red",
        )

    g = cur.gauges.get
    running = g("sglang:num_running_reqs", 0.0)
    queued = g("sglang:num_queue_reqs", 0.0)
    used = g("sglang:kv_used_tokens", 0.0)
    evict = g("sglang:kv_evictable_tokens", 0.0)
    avail = g("sglang:kv_available_tokens", 0.0)
    total = g("sglang:max_total_num_tokens", 0.0) or 1.0
    hit = g("sglang:cache_hit_rate", 0.0)

    def row(label, value, style="white"):
        t.add_row(Text(label, style=C_DIM), Text(value, style=style))

    row("running", f"{running:.0f}", C_BLUE)
    row("queued", f"{queued:.0f}", C_ORANGE if queued else C_DIM)
    # sglang:cache_hit_rate is a scheduler gauge that decays to 0 between log
    # intervals, so on its own it reads 0.0% during real traffic. Show the
    # windowed rate alongside it -- same expression the Grafana panel uses.
    hits = sum(
        rate(cur, prev, "sglang:prefill_effective_tokens_total", m) or 0.0
        for m in ("device_hit", "host_hit", "storage_hit")
    )
    allt = hits + (
        rate(cur, prev, "sglang:prefill_effective_tokens_total", "input") or 0.0
    )
    win = (hits / allt * 100) if allt > 0.001 else None
    row("cache hit (cumulative)", f"{hit * 100:.1f}%", C_MAGENTA)
    row("cache hit (windowed)", "—" if win is None else f"{win:.1f}%", C_MAGENTA)
    t.add_row("", "")
    # KV pool as a stacked bar: used | evictable | free. Kept narrow so it fits
    # the value column without forcing the panel wider.
    w = 16
    u = int(used / total * w)
    e = max(0, min(w - u, int(evict / total * w)))
    bar = Text()
    bar.append("█" * u, style=C_BLUE)
    bar.append("█" * e, style=C_YELLOW)
    bar.append("█" * max(0, w - u - e), style=C_DIM)
    t.add_row(Text("KV pool", style="bold"), bar)
    row("used", f"{used:,.0f}", C_BLUE)
    row("evictable", f"{evict:,.0f}", C_YELLOW)
    row("available", f"{avail:,.0f}", C_DIM)
    row("capacity", f"{total:,.0f}", C_DIM)
    row("utilization", f"{used / total * 100:.2f}%", C_AQUA)
    return Panel(t, title="[bold]Scheduler[/bold]", border_style=C_DIM)


def panel_throughput(
    cur: Optional[Sample], prev: Optional[Sample], hist: Dict[str, collections.deque]
) -> Panel:
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(ratio=2, no_wrap=True, overflow="ellipsis")
    t.add_column(justify="right", no_wrap=True, width=9)
    t.add_column(ratio=2, no_wrap=True, overflow="crop")

    if cur is None:
        return Panel(
            Text("—", style="red"), title="[bold]Throughput[/bold]", border_style="red"
        )

    rows = [
        ("decode", rate(cur, prev, "sglang:realtime_tokens_total", "decode"), C_BLUE),
        (
            "prefill (compute)",
            rate(cur, prev, "sglang:realtime_tokens_total", "prefill_compute"),
            C_ORANGE,
        ),
        (
            "prefill (cache hit)",
            rate(cur, prev, "sglang:realtime_tokens_total", "prefill_cache"),
            C_AQUA,
        ),
    ]
    t.add_row(
        Text("tokens/s", style="bold"),
        Text("now", style="bold"),
        Text("last 30s", style="bold"),
    )
    for label, val, color in rows:
        hist[label].append(val)
        t.add_row(
            Text(label, style=color),
            Text("—" if val is None else f"{val:,.1f}", style=color),
            Text.from_markup(spark(hist[label])),
        )
    t.add_row("", "", "")
    rr = rate(cur, prev, "sglang:num_requests_total")
    hist["req"].append(rr)
    t.add_row(
        Text("requests/s", style=C_VIOLET),
        Text("—" if rr is None else f"{rr:.2f}", style=C_VIOLET),
        Text.from_markup(spark(hist["req"])),
    )
    gt = cur.gauges.get("sglang:gen_throughput")
    t.add_row(
        Text("gen_throughput (sched)", style=C_DIM),
        Text("—" if gt is None else f"{gt:,.1f}", style=C_DIM),
        "",
    )
    return Panel(t, title="[bold]Throughput[/bold]", border_style=C_DIM)


def panel_latency(cur: Optional[Sample]) -> Panel:
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for _ in range(3):
        t.add_column(justify="right", no_wrap=True, width=9)
    t.add_row(
        Text("", style="bold"),
        Text("p50", style="bold"),
        Text("p95", style="bold"),
        Text("p99", style="bold"),
    )
    if cur is None:
        return Panel(
            Text("—", style="red"), title="[bold]Latency[/bold]", border_style="red"
        )
    # Short labels: the full names wrap to three lines in a half-width panel.
    for label, base, color in (
        ("TTFT", "sglang:time_to_first_token_seconds", C_BLUE),
        ("inter-token", "sglang:inter_token_latency_seconds", C_AQUA),
        ("end-to-end", "sglang:e2e_request_latency_seconds", C_ORANGE),
        ("queue", "sglang:queue_time_seconds", C_VIOLET),
    ):
        t.add_row(
            Text(label, style=color),
            *[Text(fmt_secs(hist_quantile(cur, base, q))) for q in (0.5, 0.95, 0.99)],
        )
    t.add_row("", "", "", "")
    n = cur.hist_count.get("sglang:e2e_request_latency_seconds")
    t.add_row(
        Text("requests observed", style=C_DIM),
        Text("—" if n is None else f"{n:,.0f}", style=C_DIM),
        "",
        "",
    )
    return Panel(
        t, title="[bold]Latency (cumulative since start)[/bold]", border_style=C_DIM
    )


def panel_ipc(procs: List[ProcInfo], socks: Dict[int, List[str]]) -> Panel:
    body = Table.grid(padding=(0, 1), expand=True)
    body.add_column(ratio=1)
    diagram = Text()
    diagram.append("  HTTP :30000\n", style=C_DIM)
    diagram.append("       │\n", style=C_DIM)
    diagram.append("  ┌────▼─────────────────────────┐\n", style=C_DIM)
    diagram.append("  │ TokenizerManager (main proc) │\n", style=C_BLUE)
    diagram.append("  └────┬─────────────────────▲───┘\n", style=C_DIM)
    diagram.append("   zmq │ ipc://              │ ipc://\n", style=C_DIM)
    diagram.append("  ┌────▼───────┐      ┌──────┴──────────┐\n", style=C_DIM)
    diagram.append("  │ Scheduler  │─────▶│ Detokenizer     │\n", style=C_ORANGE)
    diagram.append("  │ +TpWorker  │ zmq  │ Manager         │\n", style=C_ORANGE)
    diagram.append("  │ +ModelRunr │      └─────────────────┘\n", style=C_ORANGE)
    diagram.append("  └────────────┘\n", style=C_DIM)
    body.add_row(diagram)

    tbl = Table.grid(padding=(0, 1), expand=True)
    tbl.add_column(justify="right", style=C_DIM, no_wrap=True)
    tbl.add_column(ratio=1)
    n = 0
    for p in procs:
        for path in socks.get(p.pid, []):
            tbl.add_row(str(p.pid), Text(os.path.basename(path), style=C_AQUA))
            n += 1
    if n == 0:
        tbl.add_row("", Text("no ipc:// sockets visible via lsof", style=C_DIM))
    body.add_row(Text("bound unix sockets", style="bold"))
    body.add_row(tbl)
    return Panel(body, title="[bold]IPC topology[/bold]", border_style=C_DIM)


def panel_requests(tail: RequestTail) -> Panel:
    t = Table.grid(padding=(0, 2), expand=True)
    t.add_column(style=C_DIM, no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(justify="right", no_wrap=True)
    t.add_column(justify="right", style=C_DIM, no_wrap=True)
    t.add_row(
        Text("TIME", style="bold"),
        Text("RID", style="bold"),
        Text("DUR", style="bold"),
        Text("MAXTOK", style="bold"),
    )
    if not tail.events:
        t.add_row("", Text("no requests yet", style=C_DIM), "", "")
    for ev in list(tail.events)[:9]:
        d = ev["dur"]
        if d is None:
            cell = Text("—", style=C_DIM)
        else:
            style = C_AQUA if d < 1 else (C_YELLOW if d < 5 else C_ORANGE)
            cell = Text(f"{d:.2f}s", style=style)
        t.add_row(ev["t"], ev["rid"], cell, str(ev["max_new"] or "—"))
    sub = Text(f"\n{tail.skipped_health} health-check requests filtered", style=C_DIM)
    return Panel(
        Group(t, sub), title="[bold]Recent requests[/bold]", border_style=C_DIM
    )


def build(procs, socks, cur, prev, hist, tail, url, err) -> Layout:
    root = Layout()
    # Fixed sizes for the top two rows so their content is never clipped; the
    # request/IPC row absorbs whatever height is left. With ratio=1 everywhere a
    # short terminal silently truncated the tallest panels.
    root.split_column(
        Layout(name="head", size=3),
        Layout(name="r1", size=11),
        Layout(name="r2", size=10),
        Layout(name="r3", ratio=1, minimum_size=15),
        Layout(name="foot", size=1),
    )
    status = Text()
    status.append("SGLang Runtime TUI", style=f"bold {C_BLUE}")
    status.append(f"   {url}   ", style=C_DIM)
    if err:
        status.append("● UNREACHABLE", style="bold red")
    else:
        status.append("● live", style=f"bold {C_AQUA}")
    sched = next((p for p in procs if "Scheduler" in p.role), None)
    if sched:
        status.append(f"   scheduler up {fmt_dur(sched.uptime)}", style=C_DIM)
    root["head"].update(Panel(Align.center(status), border_style=C_DIM))

    root["r1"].split_row(Layout(name="proc", ratio=3), Layout(name="sched", ratio=2))
    root["r1"]["proc"].update(panel_processes(procs))
    root["r1"]["sched"].update(panel_scheduler(cur, prev))

    root["r2"].split_row(Layout(name="thru"), Layout(name="lat"))
    root["r2"]["thru"].update(panel_throughput(cur, prev, hist))
    root["r2"]["lat"].update(panel_latency(cur))

    root["r3"].split_row(Layout(name="ipc", ratio=2), Layout(name="req", ratio=3))
    root["r3"]["ipc"].update(panel_ipc(procs, socks))
    root["r3"]["req"].update(panel_requests(tail))

    root["foot"].update(
        Align.center(
            Text(
                "q quit   ·   ctrl-c quit   ·   read-only, no writes to the server",
                style=C_DIM,
            )
        )
    )
    return root


# --------------------------------------------------------------------------- #
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    run = os.path.dirname(here)
    default_log = os.path.join(
        run, "logs", "requests", f"{os.uname().nodename.split('.')[0]}_0.log"
    )

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--request-log", default=default_log)
    ap.add_argument(
        "--once", action="store_true", help="render one frame to stdout and exit"
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="render N frames non-interactively and exit",
    )
    args = ap.parse_args()

    metrics_url = args.url.rstrip("/") + "/metrics"
    tail = RequestTail(args.request_log)
    hist: Dict[str, collections.deque] = collections.defaultdict(
        lambda: collections.deque(maxlen=30)
    )
    prev: Optional[Sample] = None
    socks: Dict[int, List[str]] = {}
    tick = 0

    # Prime psutil's cpu_percent deltas -- the first call always returns 0.0.
    for p in psutil.process_iter():
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def frame():
        nonlocal prev, socks, tick
        procs = find_procs()
        cur = scrape(metrics_url)
        tail.poll()
        # lsof is slow (~100ms+); the socket set is static, so refresh rarely.
        if tick % 15 == 0 or not socks:
            socks = ipc_sockets([p.pid for p in procs])
        lay = build(procs, socks, cur, prev, hist, tail, args.url, cur is None)
        if cur is not None:
            prev = cur
        tick += 1
        return lay

    console = Console()

    # Non-interactive modes: render and exit. Used for testing and for capturing
    # a snapshot into notes without a TTY.
    n = 1 if args.once else args.iterations
    if n:
        for i in range(n):
            if i:
                time.sleep(args.interval)
            console.print(frame())
        return 0

    # Interactive: rich Live + raw-mode single-key read for 'q'.
    fd = sys.stdin.fileno()
    isatty = sys.stdin.isatty()
    old = termios.tcgetattr(fd) if isatty else None
    try:
        if isatty:
            tty.setcbreak(fd)
        with Live(frame(), console=console, screen=True, refresh_per_second=8) as live:
            while True:
                t0 = time.time()
                while time.time() - t0 < args.interval:
                    if isatty and select.select([sys.stdin], [], [], 0.05)[0]:
                        if sys.stdin.read(1).lower() == "q":
                            return 0
                    else:
                        time.sleep(0.02)
                live.update(frame())
    except KeyboardInterrupt:
        return 0
    finally:
        if isatty and old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    sys.exit(main())
