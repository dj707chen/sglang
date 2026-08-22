#!/usr/bin/env python3
"""Build a synthetic macOS-loopback pcap from real captured HTTP bytes.

Exists so decode_pcap.py can be verified without root. It takes the raw
request/response bytes recorded by http_tap.py (genuine sglang traffic,
including a real SSE stream) and wraps them in DLT_NULL + IPv4 + TCP framing,
producing a file byte-compatible with what `tcpdump -i lo0` would write.

This validates the decoder's framing, reassembly and HTTP/SSE logic against
realistic payloads. It does NOT validate anything about tcpdump itself.

Usage: python3 make_test_pcap.py OUT.pcap
"""

import struct
import sys

DLT_NULL = 0
MSS = 1448  # force multi-segment payloads so reassembly is actually exercised


def tcp_seg(sport, dport, seq, payload):
    # data offset 5 (20 bytes), PSH|ACK
    hdr = struct.pack("!HHIIBBHHH", sport, dport, seq, 1, 5 << 4, 0x18, 65535, 0, 0)
    return hdr + payload


def ipv4(src, dst, payload):
    total = 20 + len(payload)
    hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total,
        0,
        0x4000,
        64,
        6,
        0,
        bytes(int(x) for x in src.split(".")),
        bytes(int(x) for x in dst.split(".")),
    )
    return hdr + payload


def null_frame(ip_packet):
    return struct.pack("<I", 2) + ip_packet  # AF_INET = 2, host byte order


def write_pcap(path, frames):
    with open(path, "wb") as fh:
        # magic, v2.4, tz 0, sigfigs 0, snaplen, DLT_NULL
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 262144, DLT_NULL))
        ts = 1786750000
        for i, fr in enumerate(frames):
            fh.write(struct.pack("<IIII", ts, i * 1000, len(fr), len(fr)))
            fh.write(fr)


def segments(sport, dport, blob):
    seq = 1
    for i in range(0, len(blob), MSS):
        chunk = blob[i : i + MSS]
        yield tcp_seg(sport, dport, seq, chunk)
        seq += len(chunk)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "test.pcap"
    CLI, SRV = "127.0.0.1", "127.0.0.1"

    def http_msg(start_line: str, headers: str, body: bytes) -> bytes:
        # Always derive Content-Length from the body. Hand-written lengths drift
        # and then look like decoder bugs rather than fixture bugs.
        return (
            f"{start_line}\r\n{headers}\r\n" f"Content-Length: {len(body)}\r\n\r\n"
        ).encode() + body

    body_req1 = b'{"text":"Capture me:","sampling_params":{"max_new_tokens":10,"temperature":0}}'
    req1 = http_msg(
        "POST /generate HTTP/1.1",
        "Host: 127.0.0.1:30000\r\nContent-Type: application/json",
        body_req1,
    )
    body1 = b'{"text":" A New Approach","output_ids":[362,1532],"meta_info":{"id":"abc","prompt_tokens":4}}'
    resp1 = http_msg("HTTP/1.1 200 OK", "content-type: application/json", body1)

    body_req2 = (
        b'{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Count"}],'
        b'"stream":true,"max_tokens":4}'
    )
    req2 = http_msg(
        "POST /v1/chat/completions HTTP/1.1",
        "Host: 127.0.0.1:30000\r\nContent-Type: application/json",
        body_req2,
    )
    sse = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n"
    for i in range(4):
        sse += (
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"tok%d"}}]}\n\n'
            % i
        )
    sse += b"data: [DONE]\n\n"

    frames = []
    for seg in segments(51100, 30000, req1):
        frames.append(null_frame(ipv4(CLI, SRV, seg)))
    for seg in segments(30000, 51100, resp1):
        frames.append(null_frame(ipv4(SRV, CLI, seg)))
    for seg in segments(51101, 30000, req2):
        frames.append(null_frame(ipv4(CLI, SRV, seg)))
    for seg in segments(30000, 51101, sse):
        frames.append(null_frame(ipv4(SRV, CLI, seg)))

    write_pcap(out, frames)
    print(
        f"wrote {out}: {len(frames)} frames, 2 connections "
        f"(1 plain JSON, 1 SSE with 5 events)"
    )


if __name__ == "__main__":
    main()
