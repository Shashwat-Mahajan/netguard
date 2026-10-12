"""
streaming/alert_store.py
Thread-safe in-memory cache in front of ClickHouse.

ClickHouse is the source of truth (see clickhouse_reader.py). This deque only
holds the newest items so the API can show them before the writer's batch
has been flushed, and it is refilled from ClickHouse on startup.

Keeps last 10,000 flows and last 1,000 alerts in a circular buffer.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import TYPE_CHECKING, Deque

if TYPE_CHECKING:
    from streaming.consumer import PredictionResult


def merge_recent(cached: list[dict], stored: list[dict], n: int) -> list[dict]:
    """
    Combine cache and database rows, newest first, de-duplicated by flow_id.
    Rows with an empty flow_id are never treated as duplicates.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in sorted(
        cached + stored, key=lambda r: r.get("timestamp") or 0, reverse=True
    ):
        fid = row.get("flow_id")
        if fid:
            if fid in seen:
                continue
            seen.add(fid)
        out.append(row)
        if len(out) >= n:
            break
    return out


class AlertStore:
    """
    Thread-safe circular buffer of recent flow predictions.

    Consumers write via add().
    API reads via recent_flows(), recent_alerts(), summary_stats().
    """

    def __init__(
        self,
        max_flows: int = 10_000,
        max_alerts: int = 1_000,
    ) -> None:
        self._lock = threading.RLock()
        self._flows: Deque[dict] = collections.deque(maxlen=max_flows)
        self._alerts: Deque[dict] = collections.deque(maxlen=max_alerts)
        self._label_counts: dict[str, int] = collections.defaultdict(int)
        self._total = 0
        self._start_time = time.time()

    def add(self, result: "PredictionResult") -> None:
        """Add a prediction result. Called from the consumer thread."""
        d = result.to_dict()
        with self._lock:
            self._flows.append(d)
            self._label_counts[result.label] += 1
            self._total += 1
            if result.is_threat():
                self._alerts.append(d)

    def preload(self, flows: list[dict], alerts: list[dict]) -> None:
        """
        Refill the cache from the database at startup.
        Both lists are newest-first (as the reader returns them).
        Does not touch counters: those describe this process's session.
        """
        with self._lock:
            for d in reversed(flows):
                self._flows.append(d)
            for d in reversed(alerts):
                self._alerts.append(d)

    def recent_flows(self, n: int = 50) -> list[dict]:
        """Return the n most recent flows (newest first)."""
        with self._lock:
            items = list(self._flows)
        return list(reversed(items))[:n]

    def recent_alerts(self, n: int = 20) -> list[dict]:
        """Return the n most recent alerts (newest first)."""
        with self._lock:
            items = list(self._alerts)
        return list(reversed(items))[:n]

    def summary_stats(self) -> dict:
        """Return summary statistics for the dashboard."""
        with self._lock:
            uptime = time.time() - self._start_time
            total = self._total
            counts = dict(self._label_counts)
            alert_count = len(self._alerts)

        return {
            "total_flows": total,
            "flows_per_second": round(total / max(uptime, 1), 2),
            "uptime_seconds": round(uptime, 1),
            "alert_count": alert_count,
            "label_counts": counts,
            "threat_rate": round(
                sum(v for k, v in counts.items() if k != "Benign") / max(total, 1),
                4,
            ),
        }

    def clear(self) -> None:
        with self._lock:
            self._flows.clear()
            self._alerts.clear()
            self._label_counts.clear()
            self._total = 0


# ── Global singleton used by FastAPI routers ──────────────────────────────────
alert_store = AlertStore()
