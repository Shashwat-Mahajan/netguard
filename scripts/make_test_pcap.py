"""scripts/make_test_pcap.py — synthetic pcap for testing. Usage: python scripts/make_test_pcap.py data/test.pcap"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from scapy.all import IP, TCP, Ether, wrpcap  # type: ignore


def build(t0: float) -> list:
    pkts = []

    def add(pkt, t):
        pkt.time = t
        pkts.append(pkt)

    # 1) SYN flood: 1000 SYNs on one 5-tuple in 10 ms
    for i in range(1000):
        add(
            Ether()
            / IP(src="10.9.9.9", dst="10.0.0.9")
            / TCP(sport=1024 + i, dport=80, flags="S"),
            t0 + i * 0.00001,
        )

    # 2) Port scan: one SYN per port from one host
    for i, port in enumerate(range(1, 101)):
        add(
            Ether()
            / IP(src="10.8.8.8", dst="10.0.0.9")
            / TCP(sport=50000, dport=port, flags="S"),
            t0 + 1 + i * 0.01,
        )

    # 3) Normal conversation: handshake, data, close
    c, s = "10.0.0.20", "10.0.0.9"
    t = t0 + 3
    seq = [
        (c, s, 40000, 443, "S"),
        (s, c, 443, 40000, "SA"),
        (c, s, 40000, 443, "A"),
        (c, s, 40000, 443, "PA"),
        (s, c, 443, 40000, "PA"),
        (c, s, 40000, 443, "A"),
        (c, s, 40000, 443, "FA"),
        (s, c, 443, 40000, "FA"),
        (c, s, 40000, 443, "A"),
    ]
    for i, (a, b, sp, dp, fl) in enumerate(seq):
        add(
            Ether()
            / IP(src=a, dst=b)
            / TCP(sport=sp, dport=dp, flags=fl)
            / (b"x" * 200 if "P" in fl else b""),
            t + i * 0.2,
        )
    return pkts


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/test.pcap")
    out.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(out), build(time.time()))
    print(f"wrote {out}")
