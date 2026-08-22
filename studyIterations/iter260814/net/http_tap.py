#!/usr/bin/env python3
"""Recording TCP proxy for the SGLang HTTP API — works without root.

Why this exists: capturing loopback packets needs `sudo` (/dev/bpf* is
root-only on macOS and this account is not in access_bpf). Rather than make the
whole of Phase 5 depend on an interactive password prompt, this sits between
the client and the server and records every byte in both directions at the
application layer.

    client ──▶ :30001 (this tap) ──▶ :30000 (sglang)
                    │
                    └──▶ pcap/http-<ts>.jsonl   one record per exchange
                         pcap/http-<ts>.raw     raw bytes, both directions

Point any client at the tap instead of the server:
    curl http://127.0.0.1:30001/generate ...

Compared with tcpdump this loses the TCP/IP layer (handshakes, window sizes,
retransmits) and can't see traffic that bypasses it. It gains readability,
needs no privileges, and — unlike a pcap — reassembles SSE streams with
per-chunk arrival timings, which is the interesting part for a streaming
inference server.

Usage:
    python3 http_tap.py [--listen 30001] [--target 127.0.0.1:30000] [--outdir ../pcap]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import socketserver
import sys
import threading
import time
from typing import List, Optional, Tuple

BUFSZ = 65536


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class Recorder:
    """Serializes writes from many connection threads to the two output files."""

    def __init__(self, outdir: str, stamp: str):
        os.makedirs(outdir, exist_ok=True)
        self.jsonl_path = os.path.join(outdir, f"http-{stamp}.jsonl")
        self.raw_path = os.path.join(outdir, f"http-{stamp}.raw")
        self._lock = threading.Lock()
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")
        self._raw = open(self.raw_path, "ab")
        self.exchanges = 0

    def record(self, rec: dict, raw_req: bytes, raw_resp: bytes) -> None:
        with self._lock:
            self._jsonl.write(json.dumps(rec) + "\n")
            self._jsonl.flush()
            sep = f"\n===== {rec['t_start']} {rec.get('method','?')} {rec.get('path','?')} =====\n"
            self._raw.write(sep.encode())
            self._raw.write(b"--- REQUEST ---\n" + raw_req)
            self._raw.write(b"\n--- RESPONSE ---\n" + raw_resp + b"\n")
            self._raw.flush()
            self.exchanges += 1

    def close(self) -> None:
        with self._lock:
            self._jsonl.close()
            self._raw.close()


def parse_http_head(buf: bytes) -> Tuple[Optional[dict], int]:
    """Parse request/response head. Returns (info, body_offset) or (None, -1)."""
    idx = buf.find(b"\r\n\r\n")
    if idx < 0:
        return None, -1
    head = buf[:idx].decode("latin-1")
    lines = head.split("\r\n")
    if not lines:
        return None, -1
    start = lines[0].split()
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return {"start_line": lines[0], "parts": start, "headers": headers}, idx + 4


def body_is_complete(info: dict, body: bytes) -> bool:
    """Good enough for this server: Content-Length, or chunked terminator."""
    h = info["headers"]
    if h.get("transfer-encoding", "").lower() == "chunked":
        return body.endswith(b"0\r\n\r\n")
    if "content-length" in h:
        try:
            return len(body) >= int(h["content-length"])
        except ValueError:
            return True
    return False


def is_sse(info: dict) -> bool:
    return "text/event-stream" in info["headers"].get("content-type", "")


class TapHandler(socketserver.BaseRequestHandler):
    recorder: Recorder = None  # type: ignore[assignment]
    target: Tuple[str, int] = ("127.0.0.1", 30000)
    verbose: bool = True

    def handle(self) -> None:
        client = self.request
        client.settimeout(300)
        try:
            upstream = socket.create_connection(self.target, timeout=10)
        except OSError as e:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            print(f"[tap] upstream connect failed: {e}", file=sys.stderr)
            return
        upstream.settimeout(300)

        t_start = now_iso()
        t0 = time.time()

        # --- read the request head (+ body) from the client ------------------
        req = b""
        info = None
        body_off = -1
        while True:
            if info is None:
                chunk = client.recv(BUFSZ)
                if not chunk:
                    upstream.close()
                    return
                req += chunk
                info, body_off = parse_http_head(req)
                if info is None:
                    continue
            if (
                body_is_complete(info, req[body_off:])
                or "content-length" not in info["headers"]
            ):
                break
            chunk = client.recv(BUFSZ)
            if not chunk:
                break
            req += chunk

        upstream.sendall(req)

        method = info["parts"][0] if len(info["parts"]) > 0 else "?"
        path = info["parts"][1] if len(info["parts"]) > 1 else "?"

        # --- stream the response back, timing every chunk --------------------
        resp = b""
        chunks: List[dict] = []
        rinfo = None
        rbody_off = -1
        t_first_byte = None
        while True:
            try:
                chunk = upstream.recv(BUFSZ)
            except socket.timeout:
                break
            if not chunk:
                break
            if t_first_byte is None:
                t_first_byte = time.time() - t0
            resp += chunk
            try:
                client.sendall(chunk)
            except OSError:
                break
            if rinfo is None:
                rinfo, rbody_off = parse_http_head(resp)
            # Per-chunk timing is the whole point for SSE: it shows tokens
            # arriving one at a time rather than as one opaque body.
            chunks.append({"dt": round(time.time() - t0, 4), "n": len(chunk)})
            if rinfo is not None and not is_sse(rinfo):
                if body_is_complete(rinfo, resp[rbody_off:]):
                    break

        upstream.close()
        try:
            client.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        total = time.time() - t0
        req_body = req[body_off:] if body_off > 0 else b""
        resp_body = resp[rbody_off:] if rbody_off > 0 else b""
        sse_events = resp_body.count(b"data: ") if rinfo and is_sse(rinfo) else 0

        rec = {
            "t_start": t_start,
            "method": method,
            "path": path,
            "status": (
                rinfo["parts"][1] if rinfo and len(rinfo["parts"]) > 1 else None
            ),
            "streaming": bool(rinfo and is_sse(rinfo)),
            "sse_events": sse_events,
            "ttfb_s": round(t_first_byte, 4) if t_first_byte is not None else None,
            "total_s": round(total, 4),
            "req_bytes": len(req),
            "resp_bytes": len(resp),
            "req_headers": info["headers"],
            "resp_headers": rinfo["headers"] if rinfo else {},
            "req_body": req_body.decode("utf-8", "replace")[:4000],
            "resp_body": resp_body.decode("utf-8", "replace")[:8000],
            "chunk_timings": chunks[:200],
        }
        self.recorder.record(rec, req, resp)
        if self.verbose:
            s = rec["status"]
            extra = f" sse={sse_events}" if sse_events else ""
            print(
                f"[tap] {method} {path} -> {s}  ttfb={rec['ttfb_s']}s "
                f"total={rec['total_s']}s in={len(req)}B out={len(resp)}B{extra}",
                flush=True,
            )


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--listen", type=int, default=30001)
    ap.add_argument("--target", default="127.0.0.1:30000")
    ap.add_argument("--outdir", default=os.path.join(here, "..", "pcap"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    host, port = args.target.rsplit(":", 1)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rec = Recorder(os.path.abspath(args.outdir), stamp)

    TapHandler.recorder = rec
    TapHandler.target = (host, int(port))
    TapHandler.verbose = not args.quiet

    srv = ThreadedTCPServer(("127.0.0.1", args.listen), TapHandler)
    print(f"[tap] listening 127.0.0.1:{args.listen} -> {args.target}")
    print(f"[tap] jsonl : {rec.jsonl_path}")
    print(f"[tap] raw   : {rec.raw_path}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        rec.close()
        print(f"\n[tap] recorded {rec.exchanges} exchanges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
