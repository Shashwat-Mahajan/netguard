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


def test_syn_flood_fires_once():
    e = SignatureEngine(synflood_flows=200)
    alerts = [a for i in range(1000) for a in e.observe(_f(100 + i * 0.001))]
    assert [a.rule for a in alerts] == ["syn_flood"]
    assert alerts[0].src_ip == "10.9.9.9" and alerts[0].label == "DoS"


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
