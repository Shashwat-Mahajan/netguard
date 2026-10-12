"""
streaming/clickhouse_reader.py
Read side of the persistent store. ClickHouse is the source of truth;
AlertStore's deque is only a cache in front of it.

Degrades gracefully: if ClickHouse is down, every query returns [] and the
API falls back to the in-memory cache. Reconnects are retried at most every
30 seconds so a dead server isn't hammered.
"""

from __future__ import annotations

import threading
import time

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

_RETRY_SECONDS = 30

# Order must match _KEYS below.
_SELECT = """
SELECT flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
       toUnixTimestamp(timestamp), duration,
       fwd_packet_count, bwd_packet_count,
       total_fwd_bytes, total_bwd_bytes,
       flow_bytes_per_sec, flow_packets_per_sec,
       syn_flag_count, fin_flag_count, rst_flag_count, psh_flag_count,
       label, label_encoded, confidence,
       is_anomaly, anomaly_score, is_threat, inference_ms
FROM netguard.flows
"""

_KEYS = (
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "timestamp",
    "duration",
    "fwd_packet_count",
    "bwd_packet_count",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "syn_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "label",
    "label_encoded",
    "confidence",
    "is_anomaly",
    "anomaly_score",
    "is_threat",
    "inference_ms",
)


def _row_to_dict(row: tuple) -> dict:
    d = dict(zip(_KEYS, row))
    d["timestamp"] = float(d["timestamp"])
    d["is_anomaly"] = bool(d["is_anomaly"])
    d["is_threat"] = bool(d["is_threat"])
    d["class_probs"] = {}  # not persisted
    return d


class ClickHouseReader:
    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()  # clickhouse-driver clients are not thread-safe
        self._last_attempt = 0.0

    def connect(self, force: bool = False) -> bool:
        """Try to connect. Returns True if a working client exists."""
        with self._lock:
            return self._ensure(force)

    def _ensure(self, force: bool = False) -> bool:
        if self._client is not None:
            return True
        now = time.time()
        if not force and now - self._last_attempt < _RETRY_SECONDS:
            return False
        self._last_attempt = now
        try:
            from clickhouse_driver import Client  # type: ignore

            client = Client(
                host=settings.clickhouse_host,
                port=settings.clickhouse_port,
                database=settings.clickhouse_db,
                user=settings.clickhouse_user,
                password=settings.clickhouse_password,
                connect_timeout=3,
                send_receive_timeout=10,
            )
            client.execute("SELECT 1")
            self._client = client
            log.info("clickhouse.reader.connected")
            return True
        except Exception as exc:
            log.warning("clickhouse.reader.unavailable", error=str(exc))
            self._client = None
            return False

    def _query(self, sql: str, params: dict | None = None) -> list[tuple]:
        with self._lock:
            if not self._ensure():
                return []
            try:
                return self._client.execute(sql, params or {})
            except Exception as exc:
                log.warning("clickhouse.reader.query_failed", error=str(exc))
                self._client = None  # force reconnect on next call
                return []

    def recent_flows(self, n: int = 50) -> list[dict]:
        rows = self._query(
            _SELECT + " ORDER BY timestamp DESC LIMIT %(n)s", {"n": int(n)}
        )
        return [_row_to_dict(r) for r in rows]

    def recent_alerts(self, n: int = 20) -> list[dict]:
        rows = self._query(
            _SELECT + " WHERE is_threat = 1 ORDER BY timestamp DESC LIMIT %(n)s",
            {"n": int(n)},
        )
        return [_row_to_dict(r) for r in rows]


ch_reader = ClickHouseReader()
