"""
scripts/lab/diagnose.py
Looks inside a lab pcap: flow counts, how many flows match the SYN-only
rule condition, and the ML label breakdown. Writes nothing to ClickHouse.

  python scripts/lab/diagnose.py data/lab/flood.pcap
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from capture.pipeline import replay_pcap  # noqa: E402
from streaming.consumer import InferenceEngine  # noqa: E402


class Collector:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.flows = 0
        self.syn_only = 0
        self.has_syn = 0
        self.answered = 0
        self.dsts: Counter = Counter()
        self.labels: Counter = Counter()
        self.anomalies = 0
        self.max_conf: dict[str, float] = {}
        self.times: list[float] = []

    def handle_flow(self, flow) -> None:
        self.flows += 1
        syn = flow.syn_flag_count >= 1
        self.has_syn += syn
        if flow.bwd_packet_count > 0:
            self.answered += 1
        if syn and flow.ack_flag_count == 0 and flow.bwd_packet_count == 0:
            self.syn_only += 1
        self.dsts[(flow.dst_ip, flow.dst_port)] += 1
        self.times.append(float(flow.timestamp))
        r = self.engine.predict(flow)
        if r is None:
            return
        self.labels[r.label] += 1
        self.anomalies += bool(r.is_anomaly)
        self.max_conf[r.label] = max(
            self.max_conf.get(r.label, 0.0), float(r.confidence)
        )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/lab/diagnose.py <pcap>")
        return 1
    engine = InferenceEngine()
    engine.load()
    c = Collector(engine)
    n = replay_pcap(sys.argv[1], c)
    print(f"\npackets: {n}   flows: {c.flows}")
    print(f"flows with a SYN flag:            {c.has_syn}")
    print(f"flows the victim answered (bwd>0): {c.answered}")
    print(f"flows matching the SYN-only rule:  {c.syn_only}  (rule needs 200 in 10 s)")
    if c.times:
        print(f"flow timestamps span: {max(c.times) - min(c.times):.1f} s")
    print("top targets:", c.dsts.most_common(3))
    print("ML labels:", dict(c.labels))
    print(
        "ML max confidence per label:", {k: round(v, 3) for k, v in c.max_conf.items()}
    )
    print(f"Isolation Forest anomalies: {c.anomalies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
