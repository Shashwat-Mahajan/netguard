"""tests/test_pipeline.py — pipeline logic and pcap replay, no models or ClickHouse needed."""

from __future__ import annotations

import time
from types import SimpleNamespace

from capture.pipeline import FlowPipeline, replay_pcap


class StubEngine:
    def __init__(self, none=False):
        self.none = none

    def predict(self, flow):
        if self.none:
            return None
        return SimpleNamespace(flow=flow, is_threat=lambda: flow.syn_flag_count > 50)


class StubWriter:
    def __init__(self):
        self.rows = []

    def write(self, r):
        self.rows.append(r)


def _flow(**kw):
    return SimpleNamespace(timestamp=1.0, syn_flag_count=kw.get("syn", 0))


def test_handle_flow_scores_and_writes():
    w = StubWriter()
    p = FlowPipeline(StubEngine(), writer=w)
    p.handle_flow(_flow(syn=100))
    p.handle_flow(_flow(syn=0))
    assert p.summary == {"flows_seen": 2, "scored": 2, "threats": 1, "dropped": 0}
    assert len(w.rows) == 2


def test_missing_models_drop_flows_without_crashing():
    w = StubWriter()
    p = FlowPipeline(StubEngine(none=True), writer=w)
    p.handle_flow(_flow())
    assert p.summary["dropped"] == 1 and w.rows == []


def test_restamp_sets_current_time():
    p = FlowPipeline(StubEngine(), restamp=True)
    f = _flow()
    p.handle_flow(f)
    assert abs(f.timestamp - time.time()) < 5


def test_pcap_replay_uses_packet_time(tmp_path):
    from scapy.all import IP, TCP, Ether, wrpcap  # type: ignore

    t0 = 1_700_000_000.0
    pkts = []
    for i in range(200):
        pk = (
            Ether()
            / IP(src="10.9.9.9", dst="10.0.0.9")
            / TCP(sport=4444, dport=80, flags="S")
        )
        pk.time = t0 + i * 0.001
        pkts.append(pk)
    path = tmp_path / "flood.pcap"
    wrpcap(str(path), pkts)

    w = StubWriter()
    p = FlowPipeline(StubEngine(), writer=w)
    n = replay_pcap(path, p)

    assert n == 200
    assert p.summary["flows_seen"] == 1 and p.summary["threats"] == 1
    f = w.rows[0].flow
    assert f.syn_flag_count == 200
    assert 0.15 < f.duration < 0.25  # pcap time, not wall-clock replay time
    assert abs(f.timestamp - t0) < 1  # kept original time (no restamp)
