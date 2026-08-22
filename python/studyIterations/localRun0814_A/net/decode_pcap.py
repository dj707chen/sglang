#!/usr/bin/env python3
"""Decode a loopback pcap of SGLang HTTP traffic — no tshark required.

`brew install wireshark` is a large install for a decode that is ~200 lines, so
this reads the pcap directly: pcap/pcapng container, macOS NULL/loopback and
Ethernet link layers, IPv4/IPv6, TCP reassembly by stream, then HTTP framing
including SSE event splitting.

Deliberately narrow. It assumes loopback capture of plaintext HTTP, ignores
retransmits and out-of-order segments beyond sequence-number sorting, and does
not do TLS. That is enough for `tcpdump -i lo0 'tcp port 30000'`, which is what
bin/capture_http.sh produces.

Usage:
    python3 decode_pcap.py FILE.pcap [--bodies] [--max-body N] [--port 30000]
"""

from __future__ import annotations

import argparse
import collections
import struct
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- pcap reader


class PcapError(Exception):
    pass


def read_packets(path: str):
    """Yield (timestamp, link_type, raw_frame). Handles pcap and pcapng."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 4:
        raise PcapError("file too short")
    magic = data[:4]
    if magic == b"\x0a\x0d\x0d\x0a":
        yield from _read_pcapng(data)
    elif magic in (
        b"\xd4\xc3\xb2\xa1",
        b"\xa1\xb2\xc3\xd4",
        b"\x4d\x3c\xb2\xa1",
        b"\xa1\xb2\x3c\x4d",
    ):
        yield from _read_pcap(data, magic)
    else:
        raise PcapError(f"unrecognised magic {magic.hex()}")


def _read_pcap(data: bytes, magic: bytes):
    little = magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1")
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    e = "<" if little else ">"
    # pcap global header: magic, vmajor, vminor, thiszone, sigfigs, snaplen, network
    _, _, _, _, _, _, link = struct.unpack(e + "IHHiIII", data[:24])
    off = 24
    n = len(data)
    while off + 16 <= n:
        ts_s, ts_frac, caplen, _ = struct.unpack(e + "IIII", data[off : off + 16])
        off += 16
        frame = data[off : off + caplen]
        off += caplen
        ts = ts_s + (ts_frac / 1e9 if nano else ts_frac / 1e6)
        yield ts, link, frame


def _read_pcapng(data: bytes):
    off, n = 0, len(data)
    e = "<"
    link = 1
    while off + 12 <= n:
        btype, blen = struct.unpack(e + "II", data[off : off + 8])
        if btype == 0x0A0D0D0A:  # SHB — byte order magic decides endianness
            bom = data[off + 8 : off + 12]
            e = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            btype, blen = struct.unpack(e + "II", data[off : off + 8])
        if blen < 12 or off + blen > n:
            break
        body = data[off + 8 : off + blen - 4]
        if btype == 0x00000001:  # IDB
            link = struct.unpack(e + "H", body[:2])[0]
        elif btype == 0x00000006:  # EPB
            _, tsh, tsl, caplen, _ = struct.unpack(e + "IIIII", body[:20])
            frame = body[20 : 20 + caplen]
            ts = ((tsh << 32) | tsl) / 1e6
            yield ts, link, frame
        off += blen


# ------------------------------------------------------------- link/IP layers


def parse_frame(
    link: int, frame: bytes
) -> Optional[Tuple[str, int, str, int, bytes, int]]:
    """-> (src_ip, src_port, dst_ip, dst_port, payload, seq) for TCP, else None."""
    if link == 0:  # DLT_NULL — macOS loopback: 4-byte address-family header
        if len(frame) < 4:
            return None
        af = struct.unpack("<I", frame[:4])[0]
        rest, ipver = frame[4:], (4 if af == 2 else 6 if af in (24, 28, 30) else None)
        if ipver is None:
            return None
    elif link == 1:  # Ethernet
        if len(frame) < 14:
            return None
        et = struct.unpack("!H", frame[12:14])[0]
        rest = frame[14:]
        ipver = 4 if et == 0x0800 else 6 if et == 0x86DD else None
        if ipver is None:
            return None
    else:
        return None

    if ipver == 4:
        if len(rest) < 20:
            return None
        ihl = (rest[0] & 0x0F) * 4
        proto = rest[9]
        src = ".".join(str(b) for b in rest[12:16])
        dst = ".".join(str(b) for b in rest[16:20])
        tcp = rest[ihl:]
    else:
        if len(rest) < 40:
            return None
        proto = rest[6]
        src = ":".join(f"{rest[8+i]:02x}{rest[9+i]:02x}" for i in range(0, 16, 2))
        dst = ":".join(f"{rest[24+i]:02x}{rest[25+i]:02x}" for i in range(0, 16, 2))
        tcp = rest[40:]

    if proto != 6 or len(tcp) < 20:
        return None
    sport, dport, seq = struct.unpack("!HHI", tcp[:8])
    doff = (tcp[12] >> 4) * 4
    return src, sport, dst, dport, tcp[doff:], seq


# --------------------------------------------------------------- HTTP framing


def dechunk(body: bytes) -> bytes:
    """Undo Transfer-Encoding: chunked.

    Without this, SSE responses come out with hex chunk-size prefixes
    interleaved between events ("141\\r\\ndata: {...}"), which reads as corrupt
    output rather than as framing.
    """
    out, i, n = bytearray(), 0, len(body)
    while i < n:
        j = body.find(b"\r\n", i)
        if j < 0:
            break
        size_field = body[i:j].split(b";")[0].strip()
        try:
            size = int(size_field, 16)
        except ValueError:
            # Not actually chunked, or we lost sync: return what we have plus
            # the remainder rather than silently truncating.
            return bytes(out) + body[i:]
        if size == 0:
            break
        out += body[j + 2 : j + 2 + size]
        i = j + 2 + size + 2  # skip the chunk data and its trailing CRLF
    return bytes(out)


def split_http_messages(blob: bytes) -> List[bytes]:
    """Split a reassembled direction into individual HTTP messages."""
    out, i = [], 0
    while True:
        j = blob.find(b"\r\n\r\n", i)
        if j < 0:
            if blob[i:].strip():
                out.append(blob[i:])
            break
        head = blob[i:j].decode("latin-1")
        headers = {}
        for ln in head.split("\r\n")[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        body_start = j + 4
        if "content-length" in headers:
            try:
                end = body_start + int(headers["content-length"])
            except ValueError:
                end = len(blob)
        elif headers.get("transfer-encoding", "").lower() == "chunked":
            k = blob.find(b"0\r\n\r\n", body_start)
            end = (k + 5) if k >= 0 else len(blob)
        elif "text/event-stream" in headers.get("content-type", ""):
            end = len(blob)
        else:
            end = body_start
        out.append(blob[i:end])
        if end <= i:
            break
        i = end
        if i >= len(blob):
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pcap")
    ap.add_argument(
        "--port",
        type=int,
        default=30000,
        help="server port, used to label direction (default 30000)",
    )
    ap.add_argument("--bodies", action="store_true", help="print message bodies")
    ap.add_argument("--max-body", type=int, default=600)
    ap.add_argument(
        "--summary",
        action="store_true",
        help="one line per connection with request counts, no detail",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="skip messages whose start line contains SUBSTR "
        "(repeatable). Use --exclude /metrics --exclude /health "
        "to drop Prometheus scrape noise.",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="keep only messages whose start line contains SUBSTR",
    )
    args = ap.parse_args()

    def keep(start_line: str) -> bool:
        if args.only and not any(s in start_line for s in args.only):
            return False
        return not any(s in start_line for s in args.exclude)

    # stream key -> {seq: payload}, deduplicated by sequence number so
    # retransmits collapse instead of duplicating bytes.
    streams: Dict[Tuple, Dict[int, bytes]] = collections.defaultdict(dict)
    first_ts: Dict[Tuple, float] = {}
    npkts = 0

    try:
        for ts, link, frame in read_packets(args.pcap):
            p = parse_frame(link, frame)
            if p is None:
                continue
            src, sport, dst, dport, payload, seq = p
            npkts += 1
            if not payload:
                continue
            key = (src, sport, dst, dport)
            streams[key][seq] = payload
            first_ts.setdefault(key, ts)
    except PcapError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"packets: {npkts}   tcp streams with payload: {len(streams)}\n")

    # Pair up the two directions of each connection.
    conns: Dict[Tuple, Dict[str, Tuple]] = {}
    for key in streams:
        src, sport, dst, dport = key
        if dport == args.port:
            cid = (src, sport, dst, dport)
            conns.setdefault(cid, {})["req"] = key
        elif sport == args.port:
            cid = (dst, dport, src, sport)
            conns.setdefault(cid, {})["resp"] = key

    grand = collections.Counter()

    for cid in sorted(conns, key=lambda c: first_ts.get(conns[c].get("req", c), 0)):
        parts = conns[cid]
        cli = f"{cid[0]}:{cid[1]}"

        # Decode both directions first so summary mode can count without printing.
        decoded = {}
        for which in ("req", "resp"):
            k = parts.get(which)
            if not k:
                decoded[which] = []
                continue
            blob = b"".join(streams[k][s] for s in sorted(streams[k]))
            rows = []
            for msg in split_http_messages(blob):
                idx = msg.find(b"\r\n\r\n")
                head = (
                    msg[:idx].decode("latin-1")
                    if idx > 0
                    else msg.decode("latin-1", "replace")
                )
                body = msg[idx + 4 :] if idx > 0 else b""
                if "transfer-encoding: chunked" in head.lower():
                    body = dechunk(body)
                rows.append((head.split("\r\n")[0], msg, body))
            decoded[which] = rows

        req_lines = [s for s, _, _ in decoded["req"]]
        for s in req_lines:
            grand[s.split(" HTTP")[0]] += 1

        if args.summary:
            counts = collections.Counter(s.split(" HTTP")[0] for s in req_lines)
            shown = ", ".join(f"{n}x {p}" for p, n in counts.most_common(4)) or "-"
            nbytes = sum(len(m) for _, m, _ in decoded["resp"])
            print(
                f"  {cli:<22} -> :{cid[3]}  {len(req_lines):>4} req  "
                f"{nbytes/1024:>8.1f} KB resp   {shown}"
            )
            continue

        printed_header = False
        for label, which in (("REQ ", "req"), ("RESP", "resp")):
            for start, msg, body in decoded[which]:
                # Filter on the paired request line so a response isn't orphaned
                # from a request that was filtered out.
                probe = (
                    start if which == "req" else (req_lines[0] if req_lines else start)
                )
                if not keep(probe):
                    continue
                if not printed_header:
                    print(f"--- connection {cli} -> {cid[2]}:{cid[3]} ---")
                    printed_header = True
                print(f"  {label} {start}   ({len(msg)} B)")
                if b"data: " in body:
                    events = [e for e in body.split(b"\n\n") if e.strip()]
                    print(f"       SSE: {len(events)} events")
                    if args.bodies:
                        for e in events[:5]:
                            print(f"         {e.decode('utf-8','replace')[:160]}")
                        if len(events) > 5:
                            print(f"         ... {len(events)-5} more")
                elif args.bodies and body.strip():
                    txt = body.decode("utf-8", "replace")[: args.max_body]
                    print(f"       body: {txt}")
        if printed_header:
            print()

    print("=== all requests seen ===")
    for path, n in grand.most_common():
        print(f"  {n:>5}x  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
