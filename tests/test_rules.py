"""tests/test_rules.py — signature rules, no services needed."""

from __future__ import annotations

from types import SimpleNamespace

from capture.pipeline import FlowPipeline
from capture.rules import SignatureEngine


def _f(ts, src="10.9.9.9", dst="10.0.0.9", dport=80, syn_only=True):
    return SimpleNamespace(
        timestamp=ts,
        src_ip=src,
        dst_ip=dst,
        dst_port=dport,
        syn_flag_count=1,
        ack_flag_count=0 if syn_only else 1,
        bwd_packet_count=0 if syn_only else 3,
    )


def _answered(ts, dport=80, src="10.9.9.9", dst="10.0.0.9"):
    """Half-open flow as seen in a real flood: SYN, SYN-ACK, attacker RST."""
    return SimpleNamespace(
        timestamp=ts,
        src_ip=src,
        dst_ip=dst,
        dst_port=dport,
        syn_flag_count=2,
        ack_flag_count=1,
        rst_flag_count=1,
        fin_flag_count=0,
        psh_flag_count=0,
        fwd_packet_count=2,
        bwd_packet_count=1,
    )


def test_syn_flood_fires_once():
    e = SignatureEngine(synflood_flows=200)
    alerts = [a for i in range(1000) for a in e.observe(_f(100 + i * 0.001))]
    assert [a.rule for a in alerts] == ["syn_flood"]
    assert alerts[0].src_ip == "10.9.9.9" and alerts[0].label == "DoS"


def test_flood_against_open_port_is_detected():
    e = SignatureEngine(synflood_flows=200)
    alerts = [a for i in range(1000) for a in e.observe(_answered(100 + i * 0.001))]
    assert [a.rule for a in alerts] == ["syn_flood"]


def test_scan_of_answered_ports_is_not_a_flood():
    e = SignatureEngine()
    alerts = [
        a for p in range(1, 301) for a in e.observe(_answered(100 + p * 0.01, dport=p))
    ]
    assert "syn_flood" not in [a.rule for a in alerts]


def test_port_scan_fires_but_not_flood():
    e = SignatureEngine()
    alerts = [a for p in range(1, 101) for a in e.observe(_f(100 + p * 0.01, dport=p))]
    assert [a.rule for a in alerts] == ["port_scan"]


def test_normal_traffic_is_quiet():
    e = SignatureEngine()
    alerts = [
        a
        for i in range(20)
        for a in e.observe(_f(100 + i, src="10.0.0.20", dport=443, syn_only=False))
    ]
    assert alerts == []


def test_slow_scan_outside_window_does_not_fire():
    e = SignatureEngine(window=10)
    alerts = [a for p in range(1, 101) for a in e.observe(_f(100 + p * 1.0, dport=p))]
    assert alerts == []


def test_host_sweep():
    e = SignatureEngine()
    alerts = [
        a
        for i in range(60)
        for a in e.observe(_f(100 + i * 0.05, dst=f"10.1.0.{i}", dport=22))
    ]
    assert [a.rule for a in alerts] == ["host_sweep"]


def test_pipeline_runs_rules_even_without_models():
    class NoModels:
        def predict(self, flow):
            return None

    class W:
        def __init__(self):
            self.rows = []

        def write(self, a):
            self.rows.append(a)

    w = W()
    p = FlowPipeline(
        NoModels(), rules=SignatureEngine(synflood_flows=10), alert_writer=w
    )
    for i in range(30):
        p.handle_flow(_f(100 + i * 0.001))
    assert p.summary["rules"] == {"syn_flood": 1} and len(w.rows) == 1
    assert p.summary["dropped"] == 30
