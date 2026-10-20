"""
capture/rules.py
Signature rules over a sliding window of completed flows.
Catches what per-flow ML cannot: attacks defined by a pattern across many flows.

Rules (thresholds come from settings):
  port_scan  - one source hits many distinct ports on one host
  syn_flood  - one source sends many SYN-only flows to one host
  host_sweep - one source touches many distinct hosts
Time is the flow's own timestamp, so pcap replay and live capture behave alike.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass

from config.settings import settings

_MAX_EVENTS_PER_SRC = 5000
_CLEANUP_AT = 10_000


@dataclass
class RuleAlert:
    rule: str
    label: str
    label_encoded: int
    severity: str
    src_ip: str
    dst_ip: str
    flow_count: int
    detail: str
    timestamp: float
    alert_id: str = ""
    source: str = "signature"

    def __post_init__(self) -> None:
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())


class SignatureEngine:
    def __init__(
        self,
        window: float = 10,
        portscan_ports: int = 50,
        synflood_flows: int = 200,
        sweep_hosts: int = 50,
        cooldown: float = 60,
    ) -> None:
        self.window = window
        self.portscan_ports = portscan_ports
        self.synflood_flows = synflood_flows
        self.sweep_hosts = sweep_hosts
        self.cooldown = cooldown
        self._events: dict[str, deque] = {}
        self._last: dict[tuple, float] = {}

    @classmethod
    def from_settings(cls) -> "SignatureEngine":
        return cls(
            settings.rule_window_seconds,
            settings.rule_portscan_ports,
            settings.rule_synflood_flows,
            settings.rule_sweep_hosts,
            settings.rule_cooldown_seconds,
        )

    def _cooling(self, key: tuple, ts: float) -> bool:
        last = self._last.get(key)
        if last is not None and ts - last < self.cooldown:
            return True
        self._last[key] = ts
        return False

    def observe(self, flow) -> list[RuleAlert]:
        ts = float(flow.timestamp)
        src, dst = flow.src_ip, flow.dst_ip
        if not src or not dst:
            return []
        syn_only = (
            flow.syn_flag_count >= 1
            and flow.ack_flag_count == 0
            and flow.bwd_packet_count == 0
        )

        dq = self._events.setdefault(src, deque(maxlen=_MAX_EVENTS_PER_SRC))
        while dq and dq[0][0] < ts - self.window:
            dq.popleft()
        dq.append((ts, dst, flow.dst_port, syn_only))

        if len(self._events) > _CLEANUP_AT:
            self._events = {k: v for k, v in self._events.items() if v}

        same_host = [e for e in dq if e[1] == dst]
        alerts: list[RuleAlert] = []

        ports = {e[2] for e in same_host}
        if len(ports) >= self.portscan_ports and not self._cooling(
            ("port_scan", src, dst), ts
        ):
            alerts.append(
                RuleAlert(
                    "port_scan",
                    "Probe",
                    2,
                    "high",
                    src,
                    dst,
                    len(same_host),
                    f"{len(ports)} distinct ports on {dst} within {self.window:g}s",
                    ts,
                )
            )

        syns = sum(1 for e in same_host if e[3])
        if syns >= self.synflood_flows and not self._cooling(
            ("syn_flood", src, dst), ts
        ):
            alerts.append(
                RuleAlert(
                    "syn_flood",
                    "DoS",
                    1,
                    "high",
                    src,
                    dst,
                    syns,
                    f"{syns} SYN-only flows to {dst} within {self.window:g}s",
                    ts,
                )
            )

        hosts = {e[1] for e in dq}
        if len(hosts) >= self.sweep_hosts and not self._cooling(
            ("host_sweep", src, ""), ts
        ):
            alerts.append(
                RuleAlert(
                    "host_sweep",
                    "Probe",
                    2,
                    "medium",
                    src,
                    "",
                    len(dq),
                    f"{len(hosts)} distinct hosts within {self.window:g}s",
                    ts,
                )
            )
        return alerts
