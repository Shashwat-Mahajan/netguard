"""
capture/rules.py
Signature rules over a sliding window of completed flows.
Catches what per-flow ML cannot: attacks defined by a pattern across many flows.

Rules (thresholds come from settings):
  port_scan  - one source hits many distinct ports on one host
  syn_flood  - one source sends many half-open flows to ONE port of one host
  host_sweep - one source touches many distinct hosts
A half-open flow has a SYN, no FIN/PSH, at most 2 forward packets and at most
1 reply. It covers unanswered SYNs and SYNs answered by SYN-ACK or RST.
Flows that look like a server reply (source port < 1024, destination port
>= 1024) are ignored by port_scan: when the first packet seen is the server's,
the flow direction is reversed and would look like a scan.
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
    intel: str = ""
    intel_hit: bool = False

    def __post_init__(self) -> None:
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())


def _half_open(flow) -> bool:
    if flow.syn_flag_count < 1:
        return False
    if flow.ack_flag_count == 0 and flow.bwd_packet_count == 0:
        return True  # classic unanswered SYN
    return (
        getattr(flow, "fin_flag_count", 0) == 0
        and getattr(flow, "psh_flag_count", 0) == 0
        and getattr(flow, "fwd_packet_count", 99) <= 2
        and flow.bwd_packet_count <= 1
    )


def _reply_like(flow) -> bool:
    sport = getattr(flow, "src_port", None)
    if sport is None:
        return False
    return sport < 1024 <= flow.dst_port


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
        half_open = _half_open(flow)
        reply = _reply_like(flow)

        dq = self._events.setdefault(src, deque(maxlen=_MAX_EVENTS_PER_SRC))
        while dq and dq[0][0] < ts - self.window:
            dq.popleft()
        dq.append((ts, dst, flow.dst_port, half_open, reply))

        if len(self._events) > _CLEANUP_AT:
            self._events = {k: v for k, v in self._events.items() if v}

        same_host = [e for e in dq if e[1] == dst]
        alerts: list[RuleAlert] = []

        ports = {e[2] for e in same_host if not e[4]}
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

        dport = flow.dst_port
        syns = sum(1 for e in same_host if e[3] and e[2] == dport)
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
                    f"{syns} half-open flows to {dst}:{dport} within "
                    f"{self.window:g}s",
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
