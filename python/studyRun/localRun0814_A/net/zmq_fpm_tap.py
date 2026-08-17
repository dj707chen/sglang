#!/usr/bin/env python3
"""Subscribe to SGLang's per-iteration forward-pass metrics over ZMQ.

This is the *sanctioned* way to observe scheduler internals: sglang publishes
per-iteration telemetry on a ZMQ PUB socket specifically so external consumers
can watch it without polling Prometheus. No source patching, no privileges, no
packet capture.

    Scheduler process:  SchedulerMetricsMixin._emit_forward_pass_metrics()
                          -> _FpmPublisherThread -> ZMQ PUB
    This script:        ZMQ SUB -> msgpack decode -> jsonl + stdout

(see python/sglang/srt/observability/forward_pass_metrics.py)

Wire format, read off the publisher: multipart frames
    (topic=b"", seq:uint64 big-endian, msgspec-msgpack ForwardPassMetrics)
The publisher also emits a heartbeat every 1s when idle, so silence here means
"not connected", not "server idle".

Start the server with a known endpoint first:

    ./bin/start_sglang.sh \
        --enable-forward-pass-metrics \
        --forward-pass-metrics-ipc-name ipc:///tmp/sglang-fpm

then (note sglang appends ".<dp_rank>" to the endpoint):

    python3 zmq_fpm_tap.py --endpoint ipc:///tmp/sglang-fpm.0

Usage: zmq_fpm_tap.py [--endpoint EP] [--seconds N] [--out FILE] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import msgspec
import zmq


def to_plain(obj):
    """msgspec.Struct -> nested dict, for json output."""
    if isinstance(obj, msgspec.Struct):
        return {f: to_plain(getattr(obj, f)) for f in obj.__struct_fields__}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--endpoint", default="ipc:///tmp/sglang-fpm.0")
    ap.add_argument("--seconds", type=float, default=0, help="0 = until ctrl-c")
    ap.add_argument("--out", default=os.path.join(here, "..", "pcap", "fpm.jsonl"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Decode into plain dicts rather than the ForwardPassMetrics type: this
    # script should keep working if the struct gains fields, and it must not
    # depend on importing sglang.
    decoder = msgspec.msgpack.Decoder()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 1000)
    sub.connect(args.endpoint)
    print(f"[fpm] SUB connected to {args.endpoint}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fh = open(out_path, "a", encoding="utf-8")
    print(f"[fpm] writing {out_path}")

    t0 = time.time()
    n = heartbeats = active = 0
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            try:
                frames = sub.recv_multipart()
            except zmq.Again:
                continue
            if len(frames) < 3:
                continue
            seq = int.from_bytes(frames[1], "big")
            try:
                rec = decoder.decode(frames[2])
            except msgspec.DecodeError:
                continue
            n += 1
            if isinstance(rec, dict):
                payload = rec
            else:  # msgspec encodes Structs as arrays when untyped
                payload = {"raw": to_plain(rec)}
            row = {"t": round(time.time() - t0, 4), "seq": seq, "fpm": payload}
            fh.write(json.dumps(row) + "\n")

            # Distinguish a real per-iteration emit from the 1s idle heartbeat.
            # A heartbeat is constructed with only worker_id/dp_rank set, so
            # wall_time is 0.0 and every counter is zero.
            blob = json.dumps(payload)
            sched = payload.get("scheduled_requests") or {}
            queued = payload.get("queued_requests") or {}
            has_work = any(v for v in list(sched.values()) + list(queued.values()))
            if payload.get("wall_time", 0.0) or has_work:
                active += 1
            else:
                heartbeats += 1
            if not args.quiet and n <= 400:
                print(f"[fpm] seq={seq:<6} {blob[:150]}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        sub.close(linger=0)
        ctx.term()
    dur = time.time() - t0
    print(f"\n[fpm] {n} messages in {dur:.1f}s -> {out_path}")
    print(f"[fpm]   {active} per-iteration emits, {heartbeats} idle heartbeats")
    if n and active == 0:
        print(
            "[fpm] NOTE: heartbeats only. On the MLX backend this is expected:\n"
            "[fpm]   _emit_forward_pass_metrics() returns early when wall_time == 0.0\n"
            "[fpm]   (metrics_reporter.py:985), and wall_time comes from DeviceTimer,\n"
            "[fpm]   which is built on torch.cuda.Event and is only wrapped around\n"
            "[fpm]   forwards under model_executor/ -- never under hardware_backend/mlx/.\n"
            "[fpm]   The transport works; the payload is empty on this backend."
        )
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
