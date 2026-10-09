"""
tests/test_clickhouse_writer.py
Regression tests for the row-mapping bugs in ClickHouseWriter._to_row.
No ClickHouse server needed.
"""

from __future__ import annotations

from streaming.clickhouse_writer import ClickHouseWriter
from streaming.consumer import PredictionResult


def _result(**overrides) -> PredictionResult:
    base = dict(
        flow_id="f1", src_ip="10.0.0.1", dst_ip="10.0.0.2",
        src_port=1234, dst_port=80, protocol=6,
        timestamp=1_700_000_000.0, duration=1.5,
        fwd_packet_count=10, bwd_packet_count=8,
        total_fwd_bytes=1000, total_bwd_bytes=800,
        flow_bytes_per_sec=1200.0, flow_packets_per_sec=12.0,
        syn_flag_count=3, fin_flag_count=1, rst_flag_count=0, psh_flag_count=4,
        label="Benign", label_encoded=0, confidence=0.99,
        is_anomaly=False, anomaly_score=0.1,
        class_probs={"Benign": 0.99}, inference_ms=2.5,
    )
    base.update(overrides)
    return PredictionResult(**base)


def test_row_has_25_columns():
    assert len(ClickHouseWriter._to_row(_result())) == 25


def test_benign_label_encoded_is_not_turned_into_minus_one():
    row = ClickHouseWriter._to_row(_result(label_encoded=0))
    assert row[19] == 0


def test_flow_counters_are_carried_through():
    row = ClickHouseWriter._to_row(_result())
    assert row[8:18] == (10, 8, 1000, 800, 1200.0, 12.0, 3, 1, 0, 4)


def test_threat_flag_set_for_attack():
    row = ClickHouseWriter._to_row(
        _result(label="DoS", label_encoded=1, confidence=0.95)
    )
    assert row[23] == 1
