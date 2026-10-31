"""python scripts/lab/ports.py data/lab/flood22.pcap
Shows which sources trigger port-scan-like patterns, without any services."""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from capture.pipeline import replay_pcap  # noqa: E402


class C:
    def __init__(self):
        self.d = defaultdict(set)
        self.first = {}

    def handle_flow(self, f):
        k = (f.src_ip, f.dst_ip)
        self.d[k].add(f.dst_port)
        self.first.setdefault(
            k, (f.src_port, f.dst_port, f.syn_flag_count, f.bwd_packet_count)
        )


c = C()
replay_pcap(sys.argv[1], c)
for k, ports in sorted(c.d.items(), key=lambda kv: -len(kv[1]))[:6]:
    print(k, "distinct dst ports:", len(ports), "example flow:", c.first[k])
