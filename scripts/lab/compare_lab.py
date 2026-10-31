"""
scripts/lab/compare_lab.py
Per-attack comparison on the lab pcaps: NetGuard ML, NetGuard signature rules,
Suricata with local rules, Suricata with ET Open. Writes nothing to ClickHouse
or TheHive.

  python scripts/lab/compare_lab.py
Needs Docker (Suricata). ET Open is skipped if data/et/suricata.rules is missing.

Latency here = alert time minus the attacker's first packet, both in PCAP
time (packet timestamps). It is not wall-clock live-capture latency.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from capture.pipeline import FlowPipeline, replay_pcap  # noqa: E402
from capture.rules import SignatureEngine  # noqa: E402
from streaming.consumer import InferenceEngine  # noqa: E402
from streaming.suricata import parse_eve_file  # noqa: E402

LAB = ROOT / "data" / "lab"
OUT = LAB / "out"
ATTACKER = "192.168.29.131"
PCAPS = ("scan", "flood", "sweep")
ET_RULES = ROOT / "data" / "et" / "suricata.rules"
LOCAL_RULES = ROOT / "config" / "suricata" / "local.rules"


class CountingEngine:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.labels: Counter = Counter()
        self.anomalies = 0

    def predict(self, flow):
        r = self.engine.predict(flow)
        if r is not None:
            self.labels[r.label] += 1
            self.anomalies += bool(r.is_anomaly)
        return r


class Collect:
    def __init__(self) -> None:
        self.alerts: list = []

    def write(self, a) -> None:
        self.alerts.append(a)


def first_attacker_packet(pcap: Path) -> float | None:
    from scapy.layers.inet import IP
    from scapy.layers.l2 import Ether
    from scapy.utils import PcapReader

    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            t = float(pkt.time)
            if not pkt.haslayer(IP):
                pkt = Ether(bytes(pkt))
            if pkt.haslayer(IP) and pkt[IP].src == ATTACKER:
                return t
    return None


def run_netguard(pcap: Path, engine) -> dict:
    counting = CountingEngine(engine)
    sink = Collect()
    pipe = FlowPipeline(
        counting, rules=SignatureEngine.from_settings(), alert_writer=sink
    )
    t0 = time.time()
    replay_pcap(pcap, pipe)
    return {
        "flows": pipe.seen,
        "ml_labels": dict(counting.labels),
        "ml_threats": pipe.threats,
        "ml_anomalies": counting.anomalies,
        "rule_alerts": [(a.rule, a.timestamp) for a in sink.alerts],
        "seconds": round(time.time() - t0, 1),
    }


def run_suricata(pcap: Path, rules: Path, tag: str) -> list:
    log_dir = OUT / pcap.stem / tag
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{pcap}:/pcap/in.pcap:ro",
        "-v", f"{rules.parent}:/rules:ro",
        "-v", f"{log_dir}:/logs",
        "jasonish/suricata:latest",
        "-r", "/pcap/in.pcap", "-S", f"/rules/{rules.name}",
        "-k", "none", "-l", "/logs",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    eve = log_dir / "eve.json"
    if r.returncode != 0 or not eve.is_file():
        print(f"  suricata {tag} failed on {pcap.name}: {r.stderr[-300:]}")
        return []
    return [(a.rule, a.timestamp, a.detail) for a in parse_eve_file(eve)]


def summarize(alerts: list, t0: float | None) -> str:
    if not alerts:
        return "no alert"
    names = Counter(a[0] for a in alerts)
    first = min(a[1] for a in alerts)
    lat = f", first alert +{first - t0:.1f}s" if t0 is not None else ""
    top = ", ".join(f"{k} x{v}" for k, v in names.most_common(3))
    return f"{len(alerts)} alerts ({top}){lat}"


def main() -> int:
    engine = InferenceEngine()
    engine.load()
    use_et = ET_RULES.is_file()
    if not use_et:
        print("data/et/suricata.rules not found: skipping ET Open")
    results: dict = {}
    for name in PCAPS:
        pcap = LAB / f"{name}.pcap"
        if not pcap.is_file():
            print(f"skipping {name}: {pcap} not found")
            continue
        print(f"\n=== {name} ({pcap.stat().st_size:,} bytes) ===")
        t0 = first_attacker_packet(pcap)
        ng = run_netguard(pcap, engine)
        local = run_suricata(pcap, LOCAL_RULES, "local")
        et = run_suricata(pcap, ET_RULES, "et") if use_et else None
        rules = [(r, ts, "") for r, ts in ng["rule_alerts"]]
        print(f"flows: {ng['flows']}  (replay {ng['seconds']} s)")
        print(f"ML labels: {ng['ml_labels']}  threats: {ng['ml_threats']}"
              f"  anomalies: {ng['ml_anomalies']}")  # fmt: skip
        print(f"NetGuard rules:       {summarize(rules, t0)}")
        print(f"Suricata local rules: {summarize(local, t0)}")
        if et is not None:
            print(f"Suricata ET Open:     {summarize(et, t0)}")
            for rule, _ts, detail in et[:5]:
                print(f"    ET: {detail}")
        results[name] = {
            "attacker_first_packet": t0,
            "netguard": ng,
            "suricata_local": local,
            "suricata_et": et,
        }
    out = LAB / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
