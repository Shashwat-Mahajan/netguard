"""
tests/test_capture.py
Unit tests for the capture layer.
No real network interface or Scapy required — we test feature extraction
and flow table logic directly with synthetic data.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from capture.feature_extractor import FlowFeatureExtractor, _iats, _safe_stats
from capture.sniffer import FlowTable, RawFlow, _normalise_key
from processor.feature_schema import FlowFeatures

# ── _safe_stats ────────────────────────────────────────────────────────────────


def test_safe_stats_empty():
    mean, std, mx, mn = _safe_stats([])
    assert mean == std == mx == mn == 0.0


def test_safe_stats_single():
    mean, std, mx, mn = _safe_stats([5.0])
    assert mean == 5.0
    assert mx == 5.0
    assert mn == 5.0


def test_safe_stats_multiple():
    mean, std, mx, mn = _safe_stats([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert mx == 3.0
    assert mn == 1.0


# ── _iats ──────────────────────────────────────────────────────────────────────


def test_iats_empty():
    assert _iats([]) == []


def test_iats_single():
    assert _iats([1.0]) == []


def test_iats_multiple():
    result = _iats([1.0, 1.5, 2.5])
    assert result == pytest.approx([0.5, 1.0])


# ── _normalise_key ────────────────────────────────────────────────────────────


def test_normalise_key_forward():
    key, is_fwd = _normalise_key("10.0.0.1", 1234, "10.0.0.2", 80, 6)
    assert key == ("10.0.0.1", 1234, "10.0.0.2", 80, 6)
    assert is_fwd is True


def test_normalise_key_reverse():
    # Same flow, reverse direction — should produce same key
    key1, _ = _normalise_key("10.0.0.1", 1234, "10.0.0.2", 80, 6)
    key2, _ = _normalise_key("10.0.0.2", 80, "10.0.0.1", 1234, 6)
    assert key1 == key2


# ── FlowTable ─────────────────────────────────────────────────────────────────


def test_flow_table_get_or_create():
    table = FlowTable(timeout=60)
    key = ("1.1.1.1", 100, "2.2.2.2", 80, 6)
    flow = table.get_or_create(key)
    assert isinstance(flow, RawFlow)
    assert table.get_or_create(key) is flow  # same object


def test_flow_table_eviction():
    table = FlowTable(timeout=0)  # timeout=0 → everything is stale immediately
    key = ("1.1.1.1", 100, "2.2.2.2", 80, 6)
    table.get_or_create(key)
    # Force last_seen to be far in the past
    table._flows[key].last_seen = time.time() - 999
    evicted = table.evict_timed_out()
    assert len(evicted) == 1
    assert len(table) == 0


def test_flow_table_drain():
    table = FlowTable(timeout=999)
    for i in range(5):
        table.get_or_create((f"10.0.0.{i}", i, "8.8.8.8", 80, 6))
    drained = table.drain_all()
    assert len(drained) == 5
    assert len(table) == 0


# ── FlowFeatureExtractor ──────────────────────────────────────────────────────


def _make_raw_flow(
    n_fwd: int = 5,
    n_bwd: int = 3,
    with_tcp_flags: bool = True,
) -> RawFlow:
    now = time.time()
    key = ("192.168.1.1", 54321, "8.8.8.8", 443, 6)
    raw = RawFlow(key=key, start_time=now - 1.5, last_seen=now)

    for i in range(n_fwd):
        raw.fwd_packets.append(
            {"len": 100 + i * 10, "time": now - 1.5 + i * 0.1, "header": 40}
        )
    for i in range(n_bwd):
        raw.bwd_packets.append(
            {"len": 500 + i * 20, "time": now - 1.0 + i * 0.15, "header": 40}
        )

    raw.fwd_header_len = 40 * n_fwd
    raw.bwd_header_len = 40 * n_bwd

    if with_tcp_flags:
        raw.syn_flags = 1
        raw.ack_flags = n_fwd + n_bwd
        raw.fin_flags = 1
        raw.psh_flags = 2
        raw.fwd_psh = 1
        raw.bwd_psh = 1
        raw.fwd_win = 65535
        raw.bwd_win = 8192

    return raw


def test_extractor_produces_flow_features():
    extractor = FlowFeatureExtractor()
    raw = _make_raw_flow()
    f = extractor.extract(raw)
    assert isinstance(f, FlowFeatures)


def test_extractor_packet_counts():
    extractor = FlowFeatureExtractor()
    f = extractor.extract(_make_raw_flow(n_fwd=5, n_bwd=3))
    assert f.fwd_packet_count == 5
    assert f.bwd_packet_count == 3


def test_extractor_rates_positive():
    extractor = FlowFeatureExtractor()
    f = extractor.extract(_make_raw_flow())
    assert f.flow_bytes_per_sec > 0
    assert f.flow_packets_per_sec > 0


def test_extractor_tcp_flags():
    extractor = FlowFeatureExtractor()
    f = extractor.extract(_make_raw_flow(with_tcp_flags=True))
    assert f.syn_flag_count == 1
    assert f.fin_flag_count == 1
    assert f.psh_flag_count == 2
    assert f.fwd_win_bytes == 65535
    assert f.bwd_win_bytes == 8192


def test_extractor_empty_flow():
    """An empty flow (no packets) should not raise."""
    extractor = FlowFeatureExtractor()
    raw = _make_raw_flow(n_fwd=0, n_bwd=0, with_tcp_flags=False)
    f = extractor.extract(raw)
    assert f.fwd_packet_count == 0
    assert f.flow_bytes_per_sec == pytest.approx(0.0, abs=1.0)


def test_extractor_vector_length():
    """to_vector() must match ML_FEATURES length."""
    extractor = FlowFeatureExtractor()
    f = extractor.extract(_make_raw_flow())
    vec = f.to_vector()
    assert len(vec) == len(FlowFeatures.ML_FEATURES)


def test_label_encoding():
    f = FlowFeatures(label="DoS Hulk")
    f.encode_label()
    assert f.label_encoded == 1

    f2 = FlowFeatures(label="BENIGN")
    f2.encode_label()
    assert f2.label_encoded == 0

    f3 = FlowFeatures(label="????")
    f3.encode_label()
    assert f3.label_encoded == -1
