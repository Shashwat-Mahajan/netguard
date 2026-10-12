"""
tests/test_persistence.py
Cache/merge logic (no services needed) plus one round-trip against a real
ClickHouse that is skipped when the server isn't reachable.
"""

from __future__ import annotations

import time
import uuid

import pytest

from streaming.alert_store import AlertStore, merge_recent
from streaming.clickhouse_reader import ClickHouseReader
from streaming.clickhouse_writer import ClickHouseWriter
from streaming.consumer import PredictionResult


def _row(fid, ts, **kw):
    return {"flow_id": fid, "timestamp": ts, **kw}


def test_merge_orders_newest_first_and_limits():
    cached = [_row("c1", 100), _row("c2", 90)]
    stored = [_row("s1", 95), _row("s2", 10)]
    out = merge_recent(cached, stored, 3)
    assert [r["flow_id"] for r in out] == ["c1", "s1", "c2"]


def test_merge_deduplicates_by_flow_id():
    cached = [_row("a", 100)]
    stored = [_row("a", 100), _row("b", 50)]
    out = merge_recent(cached, stored, 10)
    assert [r["flow_id"] for r in out] == ["a", "b"]


def test_merge_keeps_rows_with_empty_flow_id():
    out = merge_recent([_row("", 2), _row("", 1)], [], 10)
    assert len(out) == 2


def test_preload_fills_cache_newest_first():
    store = AlertStore()
    store.preload(
        flows=[_row("new", 3), _row("old", 1)],
        alerts=[_row("new", 3)],
    )
    assert [f["flow_id"] for f in store.recent_flows(10)] == ["new", "old"]
    assert [a["flow_id"] for a in store.recent_alerts(10)] == ["new"]


# ── Integration: needs the docker-compose.phase4.yml ClickHouse ──────────────


def _clickhouse_up() -> bool:
    return ClickHouseReader().connect(force=True)


@pytest.mark.skipif(not _clickhouse_up(), reason="ClickHouse not reachable")
def test_write_then_read_roundtrip():
    fid = f"test-{uuid.uuid4()}"
    now = time.time()
    result = PredictionResult(
        flow_id=fid,
        src_ip="203.0.113.250",
        dst_ip="203.0.113.251",
        src_port=1234,
        dst_port=80,
        protocol=6,
        timestamp=now,
        duration=1.5,
        fwd_packet_count=10,
        bwd_packet_count=8,
        total_fwd_bytes=1000,
        total_bwd_bytes=800,
        flow_bytes_per_sec=1200.0,
        flow_packets_per_sec=12.0,
        syn_flag_count=3,
        fin_flag_count=1,
        rst_flag_count=0,
        psh_flag_count=4,
        label="Benign",
        label_encoded=0,
        confidence=0.99,
        is_anomaly=False,
        anomaly_score=0.1,
        class_probs={},
        inference_ms=2.5,
    )

    writer = ClickHouseWriter()
    writer.start()
    assert writer.stats["connected"]
    writer.write(result)
    writer.stop()  # drains the queue

    reader = ClickHouseReader()
    assert reader.connect(force=True)
    rows = [r for r in reader.recent_flows(200) if r["flow_id"] == fid]
    assert len(rows) == 1
    row = rows[0]
    assert row["label_encoded"] == 0  # Benign must not become -1
    assert row["total_fwd_bytes"] == 1000  # counters must not be zeroed
    assert row["psh_flag_count"] == 4
    assert abs(row["timestamp"] - now) < 5  # no timezone shift
