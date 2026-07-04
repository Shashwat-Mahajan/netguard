"""
streaming/clickhouse_writer.py
Writes flow predictions to ClickHouse for persistent storage.

Every classified flow gets written here so:
  1. Grafana can query historical data
  2. You can run SQL analytics on your flows
  3. Research paper experiments can use stored results

Uses clickhouse-driver (native TCP protocol — much faster than HTTP).

Falls back silently if ClickHouse is not running — the API keeps
working, flows just aren't persisted.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING

from config.logger import get_logger

if TYPE_CHECKING:
    from streaming.consumer import PredictionResult

log = get_logger(__name__)

CLICKHOUSE_HOST     = "localhost"
CLICKHOUSE_PORT     = 9000
CLICKHOUSE_DB       = "netguard"
CLICKHOUSE_USER     = "netguard"
CLICKHOUSE_PASSWORD = "netguard123"

INSERT_SQL = """
INSERT INTO netguard.flows (
    flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
    timestamp, duration,
    fwd_packet_count, bwd_packet_count,
    total_fwd_bytes, total_bwd_bytes,
    flow_bytes_per_sec, flow_packets_per_sec,
    syn_flag_count, fin_flag_count, rst_flag_count, psh_flag_count,
    label, label_encoded, confidence,
    is_anomaly, anomaly_score, is_threat, inference_ms
) VALUES
"""


class ClickHouseWriter:
    """
    Async batch writer to ClickHouse.

    Flows are buffered in a queue and written in batches of 100
    or every 5 seconds — whichever comes first. This avoids
    hammering ClickHouse with one insert per flow.

    Usage:
        writer = ClickHouseWriter()
        writer.start()
        writer.write(prediction_result)
        writer.stop()
    """

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        self._batch_size    = batch_size
        self._flush_interval = flush_interval
        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._client = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._written = 0
        self._failed  = 0

    def start(self) -> None:
        """Connect to ClickHouse and start the background writer thread."""
        try:
            from clickhouse_driver import Client  # type: ignore
            self._client = Client(
                host=CLICKHOUSE_HOST,
                port=CLICKHOUSE_PORT,
                database=CLICKHOUSE_DB,
                user=CLICKHOUSE_USER,
                password=CLICKHOUSE_PASSWORD,
                connect_timeout=5,
                send_receive_timeout=10,
            )
            # Test connection
            self._client.execute("SELECT 1")
            log.info("clickhouse.connected", host=CLICKHOUSE_HOST)
        except Exception as exc:
            log.warning(
                "clickhouse.unavailable",
                error=str(exc),
                hint="Start ClickHouse: docker compose -f docker-compose.phase4.yml up -d clickhouse",
            )
            self._client = None
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._write_loop,
            daemon=True,
            name="clickhouse-writer",
        )
        self._thread.start()
        log.info("clickhouse.writer.started", batch_size=self._batch_size)

    def write(self, result: "PredictionResult") -> None:
        """Queue a result for writing. Non-blocking."""
        if self._client is None:
            return
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            log.warning("clickhouse.queue_full", dropped=1)

    def stop(self) -> None:
        self._running = False
        self._flush()
        log.info("clickhouse.writer.stopped",
                 written=self._written, failed=self._failed)

    # ── Internal ──────────────────────────────────────────────

    def _write_loop(self) -> None:
        last_flush = time.time()
        while self._running:
            time_elapsed = time.time() - last_flush
            queue_full   = self._queue.qsize() >= self._batch_size

            if queue_full or time_elapsed >= self._flush_interval:
                self._flush()
                last_flush = time.time()
            else:
                time.sleep(0.5)

    def _flush(self) -> None:
        if self._client is None or self._queue.empty():
            return

        batch = []
        try:
            while not self._queue.empty() and len(batch) < self._batch_size:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if not batch:
            return

        rows = [self._to_row(r) for r in batch]

        try:
            self._client.execute(
                """INSERT INTO netguard.flows (
                    flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                    timestamp, duration,
                    fwd_packet_count, bwd_packet_count,
                    total_fwd_bytes, total_bwd_bytes,
                    flow_bytes_per_sec, flow_packets_per_sec,
                    syn_flag_count, fin_flag_count, rst_flag_count, psh_flag_count,
                    label, label_encoded, confidence,
                    is_anomaly, anomaly_score, is_threat, inference_ms
                ) VALUES""",
                rows,
            )
            self._written += len(rows)
            log.info("clickhouse.batch_written", count=len(rows), total=self._written)
        except Exception as exc:
            self._failed += len(rows)
            log.error("clickhouse.write_error", error=str(exc), batch_size=len(rows))

    @staticmethod
    def _to_row(r: "PredictionResult") -> tuple:
        from datetime import datetime
        ts = datetime.fromtimestamp(r.timestamp) if r.timestamp else datetime.now()
        return (
            r.flow_id or "",
            r.src_ip or "",
            r.dst_ip or "",
            r.src_port or 0,
            r.dst_port or 0,
            r.protocol or 0,
            ts,
            float(r.duration or 0),
            int(r.fwd_packet_count if hasattr(r, "fwd_packet_count") else 0),
            int(r.bwd_packet_count if hasattr(r, "bwd_packet_count") else 0),
            int(r.total_fwd_bytes  if hasattr(r, "total_fwd_bytes")  else 0),
            int(r.total_bwd_bytes  if hasattr(r, "total_bwd_bytes")  else 0),
            float(r.flow_bytes_per_sec   if hasattr(r, "flow_bytes_per_sec")   else 0),
            float(r.flow_packets_per_sec if hasattr(r, "flow_packets_per_sec") else 0),
            int(r.syn_flag_count if hasattr(r, "syn_flag_count") else 0),
            int(r.fin_flag_count if hasattr(r, "fin_flag_count") else 0),
            int(r.rst_flag_count if hasattr(r, "rst_flag_count") else 0),
            int(r.psh_flag_count if hasattr(r, "psh_flag_count") else 0),
            r.label or "Unknown",
            int(r.label_encoded or -1),
            float(r.confidence or 0),
            int(r.is_anomaly or 0),
            float(r.anomaly_score or 0),
            int(r.is_threat() if callable(r.is_threat) else 0),
            float(r.inference_ms or 0),
        )

    @property
    def stats(self) -> dict:
        return {
            "written":   self._written,
            "failed":    self._failed,
            "queued":    self._queue.qsize(),
            "connected": self._client is not None,
        }


# ── Global singleton ──────────────────────────────────────────
ch_writer = ClickHouseWriter()
