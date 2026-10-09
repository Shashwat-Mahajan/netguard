"""
streaming/alert_store.py
Thread-safe in-memory store of recent predictions and alerts.

The consumer writes results here.
The FastAPI endpoints read from here.
ClickHouse (Phase 4) will replace the in-memory store for persistence.

Keeps last 10,000 flows and last 1,000 alerts in a circular buffer.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import TYPE_CHECKING, Deque

if TYPE_CHECKING:
    from streaming.consumer import PredictionResult


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
            "total_flows":     total,
            "flows_per_second": round(total / max(uptime, 1), 2),
            "uptime_seconds":  round(uptime, 1),
            "alert_count":     alert_count,
            "label_counts":    counts,
            "threat_rate":     round(
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
