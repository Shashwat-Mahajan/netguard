"""
streaming/shuffle.py
Posts NetGuard alerts as JSON to a SOAR webhook (Shuffle webhook trigger or
scripts/soar_receiver.py). Disabled unless SOAR_WEBHOOK_URL is set. Failures
are logged and swallowed. Each alert is sent once.
"""

from __future__ import annotations

import threading

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

_RANK = {"low": 1, "medium": 2, "high": 3}
_MAX_SEEN = 20_000


def _default_post(url: str, payload: dict, timeout: float) -> int:
    import httpx

    return httpx.post(url, json=payload, timeout=timeout).status_code


class SoarClient:
    def __init__(self, url="", min_severity="high", timeout=5, http_post=None):
        self._url = url
        self._min = _RANK.get(min_severity, 3)
        self._timeout = timeout
        self._post = http_post or _default_post
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self.sent = 0
        self.failed = 0

    @classmethod
    def from_settings(cls) -> "SoarClient":
        s = settings
        return cls(s.soar_webhook_url, s.soar_min_severity, s.soar_timeout_seconds)

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def send_rule_alert(self, a) -> bool:
        if not self.enabled:
            return False
        hit = bool(getattr(a, "intel_hit", False))
        if _RANK.get(a.severity, 1) < self._min and not hit:
            return False
        with self._lock:
            if a.alert_id in self._seen:
                return False
            if len(self._seen) >= _MAX_SEEN:
                self._seen.clear()
            self._seen.add(a.alert_id)
        payload = {
            "alert_id": a.alert_id,
            "source": a.source,
            "rule": a.rule,
            "label": a.label,
            "severity": a.severity,
            "src_ip": a.src_ip,
            "dst_ip": a.dst_ip,
            "flow_count": a.flow_count,
            "detail": a.detail,
            "timestamp": a.timestamp,
            "intel": getattr(a, "intel", ""),
            "intel_hit": hit,
        }
        try:
            code = self._post(self._url, payload, self._timeout)
        except Exception as exc:
            code = 0
            log.warning("soar.send_failed", error=str(exc))
        if 200 <= code < 300:
            self.sent += 1
            return True
        self.failed += 1
        with self._lock:
            self._seen.discard(a.alert_id)
        return False
