"""
streaming/clickhouse_writer.py
Writes flow predictions to ClickHouse for persistent storage.

Every classified flow gets written here so:
  1. Grafana can query historical data
  2. You can run SQL analytics on your flows
  3. Research paper experiments can use stored results

Uses clickhouse-driver (native TCP protocol — much faster than HTTP).
Connection settings come from config.settings (CLICKHOUSE_* env vars).

Falls back silently if ClickHouse is not running — the API keeps
working, flows just aren't persisted.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from config.logger import get_logger
from config.settings import settings

if TYPE_CHECKING:
    from streaming.consumer import PredictionResult

log = get_logger(__name__)

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
    Batch writer to ClickHouse.

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
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._client = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._written = 0
        self._failed = 0

    def start(self) -> None:
        """Connect to ClickHouse and start the background writer thread."""
        try:
            from clickhouse_driver import Client  # type: ignore
            self._client = Client(
                host=settings.clickhouse_host,
                port=settings.clickhouse_port,
                database=settings.clickhouse_db,
                user=settings.clickhouse_user,
                password=settings.clickhouse_password,
                connect_timeout=5,
                send_receive_timeout=10,
            )
            # Test connection
            self._client.execute("SELECT 1")
            log.info("clickhouse.connected", host=settings.clickhouse_host)
        except Exception as exc:
            log.warning(
                "clickhouse.unavailable",
                error=str(exc),
                hint="Start ClickHouse: docker compose -f docker-compose.phase4.yml up -d clickhouse",
            )
            self._client = None
            return

        self._running = True
        self._thread = threading.Thread(
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
        """Stop the loop and drain whatever is still queued."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._flush_interval + 1)
        # Drain remaining batches (bounded so shutdown can't hang forever)
        for _ in range(100):
            if self._queue.empty():
                break
            self._flush()
        log.info("clickhouse.writer.stopped",
                 written=self._written, failed=self._failed)

    # ── Internal ──────────────────────────────────────────────

    def _write_loop(self) -> None:
        last_flush = time.time()
        while self._running:
            time_elapsed = time.time() - last_flush
            queue_full = self._queue.qsize() >= self._batch_size

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
            self._client.execute(INSERT_SQL, rows)
            self._written += len(rows)
            log.info("clickhouse.batch_written", count=len(rows), total=self._written)
        except Exception as exc:
            self._failed += len(rows)
            log.error("clickhouse.write_error", error=str(exc), batch_size=len(rows))

    @staticmethod
    def _to_row(r: "PredictionResult") -> tuple:
        ts = (
            datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
            if r.timestamp
            else datetime.now(timezone.utc)
        )
        return (
            r.flow_id or "",
            r.src_ip or "",
            r.dst_ip or "",
            int(r.src_port or 0),
            int(r.dst_port or 0),
            int(r.protocol or 0),
            ts,
            float(r.duration or 0),
            int(r.fwd_packet_count),
            int(r.bwd_packet_count),
            int(r.total_fwd_bytes),
            int(r.total_bwd_bytes),
            float(r.flow_bytes_per_sec),
            float(r.flow_packets_per_sec),
            int(r.syn_flag_count),
            int(r.fin_flag_count),
            int(r.rst_flag_count),
            int(r.psh_flag_count),
            r.label or "Unknown",
            # NB: don't use `or -1` here — 0 (Benign) is falsy
            int(r.label_encoded) if r.label_encoded is not None else -1,
            float(r.confidence),
            int(bool(r.is_anomaly)),
            float(r.anomaly_score),
            int(r.is_threat()),
            float(r.inference_ms),
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
