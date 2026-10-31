"""python scripts/lab/rules_only.py data/lab/flood8080.pcap [...]
Runs only the signature rules over pcaps (no ML), so floods replay in seconds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from capture.pipeline import replay_pcap  # noqa: E402
from capture.rules import SignatureEngine  # noqa: E402


class P:
    def __init__(self):
        self.engine = SignatureEngine.from_settings()
        self.alerts = []
        self.flows = 0

    def handle_flow(self, f):
        self.flows += 1
        self.alerts += self.engine.observe(f)


for path in sys.argv[1:]:
    p = P()
    replay_pcap(path, p)
    print(f"\n{path}: {p.flows} flows")
    for a in p.alerts:
        print(f"  {a.rule}: {a.src_ip} -> {a.dst_ip}  {a.detail}")
    if not p.alerts:
        print("  no alert")
