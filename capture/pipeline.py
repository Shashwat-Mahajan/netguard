"""
capture/pipeline.py
Glue between capture and the rest of NetGuard.

For every completed flow:  signature rules (+ threat intel, TheHive) and predict ->
ClickHouse (and optionally Kafka, TheHive).
Degrades gracefully: no models -> flows are counted and dropped (rules still run),
ClickHouse down -> the writers ignore them, Kafka/intel/TheHive off -> skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from config.logger import get_logger

log = get_logger(__name__)

_QUEUE_HIGH_WATER = 9000  # ClickHouseWriter drops above 10,000 queued rows


class FlowPipeline:
    def __init__(
        self,
        engine,
        writer=None,
        store=None,
        producer=None,
        restamp: bool = False,
        rules=None,
        alert_writer=None,
        intel=None,
        thehive=None,
    ) -> None:
        self._engine = engine
        self._writer = writer
        self._store = store
        self._producer = producer
        self._restamp = restamp
        self._rules = rules
        self._alert_writer = alert_writer
        self._intel = intel
        self._thehive = thehive
        self.seen = 0
        self.scored = 0
        self.threats = 0
        self.dropped = 0
        self.rule_alerts = 0
        self.rule_counts: dict[str, int] = {}

    def _backpressure(self) -> None:
        """Replay emits flows far faster than live; don't overflow the writer queue."""
        stats = getattr(self._writer, "stats", None)
        if not isinstance(stats, dict) or not stats.get("connected"):
            return
        for _ in range(150):  # wait at most ~30 s
            if stats.get("queued", 0) < _QUEUE_HIGH_WATER:
                return
            time.sleep(0.2)
            stats = self._writer.stats

    def _run_rules(self, flow) -> None:
        try:
            alerts = self._rules.observe(flow)
        except Exception as exc:
            log.error("rules.error", error=str(exc))
            return
        for a in alerts:
            if self._restamp:
                a.timestamp = time.time()
            self.rule_alerts += 1
            self.rule_counts[a.rule] = self.rule_counts.get(a.rule, 0) + 1
            if self._intel is not None:
                try:
                    info, hit = self._intel.enrich_alert(a.src_ip, a.dst_ip)
                    a.intel = json.dumps(info)
                    a.intel_hit = hit
                except Exception as exc:
                    log.warning("intel.error", error=str(exc))
            log.warning(
                "rules.alert",
                rule=a.rule,
                src=a.src_ip,
                dst=a.dst_ip,
                severity=a.severity,
                detail=a.detail,
                intel_hit=getattr(a, "intel_hit", False),
            )
            if self._alert_writer is not None:
                self._alert_writer.write(a)
            if self._thehive is not None:
                try:
                    self._thehive.send_rule_alert(a)
                except Exception as exc:
                    log.warning("thehive.error", error=str(exc))

    def handle_flow(self, flow) -> None:
        self.seen += 1
        if self._rules is not None:
            self._run_rules(flow)  # uses the flow's original timestamp
        if self._restamp:
            flow.timestamp = time.time()  # so replayed flows show up as recent
        if self._producer is not None:
            self._producer.send(flow)
        result = self._engine.predict(flow)
        if result is None:
            self.dropped += 1
            return
        self.scored += 1
        if result.is_threat():
            self.threats += 1
            if self._thehive is not None:
                try:
                    self._thehive.send_ml_alert(result)
                except Exception as exc:
                    log.warning("thehive.error", error=str(exc))
        if self._store is not None:
            self._store.add(result)
        if self._writer is not None:
            self._backpressure()
            self._writer.write(result)

    @property
    def summary(self) -> dict:
        s = {
            "flows_seen": self.seen,
            "scored": self.scored,
            "threats": self.threats,
            "dropped": self.dropped,
        }
        if self._rules is not None:
            s["rule_alerts"] = self.rule_alerts
            s["rules"] = dict(self.rule_counts)
        if self._thehive is not None and self._thehive.enabled:
            s["thehive"] = {"sent": self._thehive.sent, "failed": self._thehive.failed}
        return s


def replay_pcap(path: str | Path, pipeline: FlowPipeline, max_packets: int = 0) -> int:
    """Read a pcap with the same flow logic as live capture. Returns packet count."""
    from scapy.layers.inet import IP  # type: ignore
    from scapy.layers.l2 import Ether  # type: ignore
    from scapy.utils import PcapReader  # type: ignore

    from capture.sniffer import PacketCapture

    cap = PacketCapture(on_flow=pipeline.handle_flow)
    n = 0
    with PcapReader(str(path)) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                # link layer wasn't decoded (no libpcap): re-parse bytes as Ethernet
                t = pkt.time
                pkt = Ether(bytes(pkt))
                pkt.time = t
            cap._handle_packet(pkt, use_pkt_time=True)
            n += 1
            if max_packets and n >= max_packets:
                break
    cap.stop()  # drains every remaining flow through the pipeline
    return n
