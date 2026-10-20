"""streaming/alert_writer.py — persists signature-rule alerts to ClickHouse."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

_CREATE = """
CREATE TABLE IF NOT EXISTS netguard.alerts (
    alert_id      String,
    timestamp     DateTime,
    source        LowCardinality(String),
    rule          LowCardinality(String),
    label         LowCardinality(String),
    label_encoded Int8,
    severity      LowCardinality(String),
    src_ip        String,
    dst_ip        String,
    flow_count    UInt32,
    detail        String,
    intel         String DEFAULT '',
    intel_hit     UInt8 DEFAULT 0
) ENGINE = MergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, src_ip)
"""

_MIGRATIONS = (
    "ALTER TABLE netguard.alerts ADD COLUMN IF NOT EXISTS intel String DEFAULT ''",
    "ALTER TABLE netguard.alerts ADD COLUMN IF NOT EXISTS intel_hit UInt8 DEFAULT 0",
)

_INSERT = (
    "INSERT INTO netguard.alerts (alert_id, timestamp, source, rule, label, "
    "label_encoded, severity, src_ip, dst_ip, flow_count, detail, intel, intel_hit) "
    "VALUES"
)


class AlertWriter:
    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()
        self.written = 0
        self.failed = 0

    def start(self) -> bool:
        try:
            from clickhouse_driver import Client  # type: ignore

            c = Client(
                host=settings.clickhouse_host,
                port=settings.clickhouse_port,
                database=settings.clickhouse_db,
                user=settings.clickhouse_user,
                password=settings.clickhouse_password,
                connect_timeout=5,
                send_receive_timeout=10,
            )
            c.execute(_CREATE)
            for sql in _MIGRATIONS:  # upgrade tables created before intel existed
                c.execute(sql)
            self._client = c
            log.info("alert_writer.started")
            return True
        except Exception as exc:
            log.warning("alert_writer.unavailable", error=str(exc))
            self._client = None
            return False

    def write(self, a) -> None:
        if self._client is None:
            return
        row = (
            a.alert_id,
            datetime.fromtimestamp(a.timestamp, tz=timezone.utc),
            a.source,
            a.rule,
            a.label,
            int(a.label_encoded),
            a.severity,
            a.src_ip,
            a.dst_ip,
            int(a.flow_count),
            a.detail,
            getattr(a, "intel", ""),
            int(bool(getattr(a, "intel_hit", False))),
        )
        with self._lock:
            try:
                self._client.execute(_INSERT, [row])
                self.written += 1
            except Exception as exc:
                self.failed += 1
                log.error("alert_writer.write_error", error=str(exc))


ch_alert_writer = AlertWriter()
