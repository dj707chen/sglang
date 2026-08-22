#!/usr/bin/env python3
"""Generate the SGLang runtime dashboard JSON.

Written as a generator rather than hand-authored JSON because the panel grid is
repetitive and easy to get subtly wrong by hand (overlapping gridPos is silent --
Grafana just stacks panels on top of each other).

Every metric name here was read off a live /metrics scrape, and every
node_exporter name off a live darwin node_exporter -- Linux names like
node_memory_MemAvailable_bytes do not exist on macOS.

Palette: the validated dark-mode categorical slots (validate_palette.js, all
checks pass on the adjacent pairlist against surface #1a1a19).

Regenerate with:  python3 build_dashboard.py > sglang-runtime.json
"""

import json

# --- Palette -----------------------------------------------------------------
# Categorical slots, dark mode. Fixed order, never cycled.
BLUE = "#3987e5"  # slot 1
ORANGE = "#d95926"  # slot 2
AQUA = "#199e70"  # slot 3
YELLOW = "#c98500"  # slot 4
MAGENTA = "#d55181"  # slot 5
VIOLET = "#9085e9"  # slot 7
# Neutral, for "absence" quantities (free/available space) -- deliberately not a
# categorical slot, because free space is not a peer category of used space.
GRAY = "#5a5a57"

# Quantiles are ordinal (p50 < p95 < p99), so they get a single-hue sequential
# ramp rather than categorical hues. Stepped for a dark surface: dim -> bright.
Q50, Q95, Q99 = "#2a5f9e", BLUE, "#8fbdf0"

DS = {"type": "prometheus", "uid": "sglang-prom"}
_id = iter(range(1, 500))


def target(expr, legend, ref="A"):
    return {
        "datasource": DS,
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend,
        "range": True,
        "refId": ref,
    }


def targets(*pairs):
    return [target(e, l, chr(ord("A") + i)) for i, (e, l) in enumerate(pairs)]


def color_overrides(mapping):
    return [
        {
            "matcher": {"id": "byName", "options": name},
            "properties": [
                {"id": "color", "value": {"fixedColor": color, "mode": "fixed"}}
            ],
        }
        for name, color in mapping.items()
    ]


def timeseries(
    title,
    x,
    y,
    w,
    h,
    tgts,
    unit="none",
    colors=None,
    desc="",
    stacked=False,
    fill=0,
    minv=None,
    maxv=None,
    axis_label="",
):
    return {
        "id": next(_id),
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": tgts,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                **({"min": minv} if minv is not None else {}),
                **({"max": maxv} if maxv is not None else {}),
                "custom": {
                    # Thin marks: 2px lines, no heavy fills.
                    "lineWidth": 2,
                    "fillOpacity": fill,
                    "showPoints": "never",
                    "spanNulls": True,
                    "axisLabel": axis_label,
                    # Recessive grid.
                    "axisBorderShow": False,
                    "gradientMode": "none",
                    "stacking": {
                        "mode": "normal" if stacked else "none",
                        "group": "A",
                    },
                    "lineInterpolation": "linear",
                },
                "color": {"mode": "palette-classic"},
            },
            "overrides": color_overrides(colors or {}),
        },
        "options": {
            # Legend always present for >=2 series; identity is never colour-alone.
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": len(tgts) >= 2,
                "calcs": [],
            },
            # Crosshair + shared tooltip: the default interaction layer.
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def stat(
    title,
    x,
    y,
    w,
    h,
    expr,
    unit="none",
    color=BLUE,
    desc="",
    decimals=None,
    thresholds=None,
):
    steps = thresholds or [{"color": color, "value": None}]
    return {
        "id": next(_id),
        "type": "stat",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, "")],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                **({"decimals": decimals} if decimals is not None else {}),
                "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute", "steps": steps},
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto",
            "textMode": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }


def row(title, y):
    return {
        "id": next(_id),
        "type": "row",
        "title": title,
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def quantiles(metric, by=""):
    """p50/p95/p99 from a Prometheus histogram."""
    grp = f"le{(', ' + by) if by else ''}"
    return [
        (
            f"histogram_quantile({q}, sum(rate({metric}_bucket[$__rate_interval])) by ({grp}))",
            lbl,
        )
        for q, lbl in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99"))
    ]


QCOLORS = {"p50": Q50, "p95": Q95, "p99": Q99}

panels = []
y = 0

# ============================================================ at a glance ====
panels.append(row("Scheduler at a glance", y))
y += 1
panels += [
    stat(
        "Running requests",
        0,
        y,
        4,
        4,
        "sglang:num_running_reqs",
        color=BLUE,
        desc="Requests currently in the running batch.",
    ),
    stat(
        "Queued requests",
        4,
        y,
        4,
        4,
        "sglang:num_queue_reqs",
        color=ORANGE,
        desc="Requests in the scheduler waiting queue. Sustained non-zero "
        "means the scheduler is the bottleneck.",
    ),
    stat(
        "KV pool used",
        8,
        y,
        4,
        4,
        "100 * sglang:kv_used_tokens / sglang:max_total_num_tokens",
        unit="percent",
        color=AQUA,
        decimals=1,
        desc="Actively used KV slots as a share of the pool "
        "(capped at 32768 tokens by --max-total-tokens).",
    ),
    stat(
        "Generation throughput",
        12,
        y,
        4,
        4,
        "sglang:gen_throughput",
        color=VIOLET,
        decimals=1,
        desc="Decode tokens/s, as reported by the scheduler.",
    ),
    stat(
        "Prefix cache hit rate",
        16,
        y,
        4,
        4,
        "100 * sglang:cache_hit_rate",
        unit="percent",
        color=MAGENTA,
        decimals=1,
        desc="Radix/prefix cache hit rate.",
    ),
    stat(
        "Scrape targets up",
        20,
        y,
        4,
        4,
        "sum(up)",
        color=AQUA,
        desc="Number of Prometheus targets currently up (expect 3: sglang, node, prometheus).",
        thresholds=[
            {"color": "#e66767", "value": None},
            {"color": YELLOW, "value": 2},
            {"color": AQUA, "value": 3},
        ],
    ),
]
y += 4

# ============================================================ request flow ===
panels.append(row("Request flow", y))
y += 1
panels += [
    timeseries(
        "Requests in flight",
        0,
        y,
        12,
        8,
        targets(
            ("sglang:num_running_reqs", "running"), ("sglang:num_queue_reqs", "queued")
        ),
        colors={"running": BLUE, "queued": ORANGE},
        fill=12,
        axis_label="requests",
        desc="Running vs queued. The gap between them is scheduler backpressure.",
    ),
    timeseries(
        "Request rate",
        12,
        y,
        12,
        8,
        targets(
            ("sum(rate(sglang:num_requests_total[$__rate_interval]))", "requests/s")
        ),
        unit="reqps",
        colors={"requests/s": BLUE},
        axis_label="req/s",
    ),
]
y += 8

# ================================================================= latency ===
panels.append(row("Latency", y))
y += 1
panels += [
    timeseries(
        "Time to first token",
        0,
        y,
        12,
        8,
        targets(*quantiles("sglang:time_to_first_token_seconds")),
        unit="s",
        colors=QCOLORS,
        axis_label="seconds",
        desc="Buckets re-floored to 0.005s via --bucket-time-to-first-token; "
        "sglang's default buckets start at 0.1s and cannot resolve this model.",
    ),
    timeseries(
        "Inter-token latency",
        12,
        y,
        12,
        8,
        targets(*quantiles("sglang:inter_token_latency_seconds")),
        unit="s",
        colors=QCOLORS,
        axis_label="seconds",
        desc="Per-token decode latency. Default buckets (0.002s floor) are "
        "already appropriate here, so they are unmodified.",
    ),
]
y += 8
panels += [
    timeseries(
        "End-to-end request latency",
        0,
        y,
        12,
        8,
        targets(*quantiles("sglang:e2e_request_latency_seconds")),
        unit="s",
        colors=QCOLORS,
        axis_label="seconds",
        desc="Buckets re-floored to 0.05s via --bucket-e2e-request-latency.",
    ),
    timeseries(
        "Queue time",
        12,
        y,
        12,
        8,
        targets(*quantiles("sglang:queue_time_seconds")),
        unit="s",
        colors=QCOLORS,
        axis_label="seconds",
        desc="Time spent waiting before the scheduler admits a request.",
    ),
]
y += 8

# ============================================================== throughput ===
panels.append(row("Throughput", y))
y += 1
panels += [
    timeseries(
        "Token throughput by mode",
        0,
        y,
        12,
        8,
        targets(
            (
                'sum(rate(sglang:realtime_tokens_total{mode="decode"}[$__rate_interval]))',
                "decode",
            ),
            (
                'sum(rate(sglang:realtime_tokens_total{mode="prefill_compute"}[$__rate_interval]))',
                "prefill (compute)",
            ),
            (
                'sum(rate(sglang:realtime_tokens_total{mode="prefill_cache"}[$__rate_interval]))',
                "prefill (cache hit)",
            ),
        ),
        colors={
            "decode": BLUE,
            "prefill (compute)": ORANGE,
            "prefill (cache hit)": AQUA,
        },
        axis_label="tokens/s",
        desc="Splits prefill into freshly computed vs served from the prefix "
        "cache. Cache-hit prefill costs almost nothing.",
    ),
    timeseries(
        "Per-stage request latency",
        12,
        y,
        12,
        8,
        targets(
            (
                "histogram_quantile(0.95, sum(rate(sglang:per_stage_req_latency_seconds_bucket[$__rate_interval])) by (le, stage))",
                "{{stage}} p95",
            ),
        ),
        unit="s",
        axis_label="seconds",
        colors={"request_process p95": BLUE, "prefill_forward p95": ORANGE},
        desc="p95 latency broken down by scheduler stage.",
    ),
]
y += 8

# ================================================================ KV cache ===
panels.append(row("KV cache & prefix reuse", y))
y += 1
panels += [
    timeseries(
        "KV pool occupancy",
        0,
        y,
        12,
        8,
        targets(
            ("sglang:kv_used_tokens", "used"),
            ("sglang:kv_evictable_tokens", "evictable (radix-cached)"),
            ("sglang:kv_available_tokens", "available"),
        ),
        stacked=True,
        fill=55,
        colors={"used": BLUE, "evictable (radix-cached)": YELLOW, "available": GRAY},
        axis_label="tokens",
        desc="Stacked to the pool total (32768). 'Evictable' is radix-cached "
        "data that can be reclaimed under pressure -- it is neither busy "
        "nor free.\n\nNOTE: sglang:kv_cache_memory_usage_gb reads 0 on the "
        "MLX backend (it measures torch-side allocation, which MLX "
        "bypasses), so this panel counts tokens, not bytes.",
    ),
    timeseries(
        "Prefix cache hit rate",
        12,
        y,
        12,
        8,
        targets(
            ("100 * sglang:cache_hit_rate", "cumulative"),
            (
                '100 * sum(rate(sglang:prefill_effective_tokens_total{mode=~".*_hit"}[$__rate_interval]))'
                " / clamp_min(sum(rate(sglang:prefill_effective_tokens_total[$__rate_interval])), 0.001)",
                "windowed",
            ),
        ),
        unit="percent",
        minv=0,
        maxv=100,
        colors={"cumulative": MAGENTA, "windowed": AQUA},
        axis_label="percent",
        desc="Cumulative (since start) vs windowed (rate over the current "
        "interval). The windowed series is what actually responds to a "
        "change in workload.",
    ),
]
y += 8

# ==================================================================== host ===
panels.append(row("Host (unified memory pressure)", y))
y += 1
panels += [
    timeseries(
        "Host CPU by mode",
        0,
        y,
        8,
        8,
        targets(
            (
                '100 * sum(rate(node_cpu_seconds_total{mode="user"}[$__rate_interval])) / count(count(node_cpu_seconds_total) by (cpu))',
                "user",
            ),
            (
                '100 * sum(rate(node_cpu_seconds_total{mode="system"}[$__rate_interval])) / count(count(node_cpu_seconds_total) by (cpu))',
                "system",
            ),
        ),
        unit="percent",
        stacked=True,
        fill=40,
        colors={"user": BLUE, "system": ORANGE},
        axis_label="percent of all cores",
    ),
    timeseries(
        "Host memory",
        8,
        y,
        8,
        8,
        targets(
            ("node_memory_wired_bytes", "wired"),
            ("node_memory_active_bytes", "active"),
            ("node_memory_compressed_bytes", "compressed"),
            ("node_memory_free_bytes", "free"),
        ),
        unit="bytes",
        stacked=True,
        fill=45,
        colors={"wired": BLUE, "active": ORANGE, "compressed": YELLOW, "free": GRAY},
        axis_label="bytes",
        desc="MLX wired memory shows up under 'wired'. Growth in 'compressed' "
        "is the early warning that the KV pool is too large for this host.",
    ),
    timeseries(
        "Swap used",
        16,
        y,
        8,
        8,
        targets(("node_memory_swap_used_bytes", "swap used")),
        unit="bytes",
        colors={"swap used": MAGENTA},
        fill=25,
        axis_label="bytes",
        desc="The unambiguous signal that memory was over-committed. Any "
        "sustained rise here invalidates latency numbers taken at the "
        "same time.",
    ),
]
y += 8

dashboard = {
    "uid": "sglang-runtime",
    "title": "SGLang Runtime — localRun0814_A",
    "description": (
        "SGLang on Apple M3 Pro via the MLX Metal backend. Every metric name "
        "here was verified against a live /metrics scrape. Panels deliberately "
        "avoid sglang:kv_cache_memory_usage_gb and sglang:weight_memory_usage_gb, "
        "which read 0 under MLX."
    ),
    "tags": ["sglang", "mlx", "study"],
    "timezone": "browser",
    "editable": True,
    "graphTooltip": 1,  # shared crosshair across panels
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "timepicker": {
        "refresh_intervals": ["1s", "5s", "10s", "30s", "1m", "5m"],
    },
    "panels": panels,
}

print(json.dumps(dashboard, indent=2))
