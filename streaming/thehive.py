"""
streaming/thehive.py
Creates TheHive 5 alerts (POST /api/v1/alert) from NetGuard alerts.

Disabled unless THEHIVE_URL and THEHIVE_API_KEY are set. Failures are logged
and swallowed, so capture never depends on TheHive. Each alert is sent once
(deduplicated by sourceRef), so replaying a pcap does not create duplicates.
"""

from __future__ import annotations

import json
import threading
import time

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)

_RANK = {"low": 1, "medium": 2, "high": 3}
_MAX_SEEN = 20_000


def _default_post(url: str, payload: dict, headers: dict, timeout: float) -> int:
    import httpx

    return httpx.post(url, json=payload, headers=headers, timeout=timeout).status_code


class TheHiveClient:
    def __init__(
        self, url="", api_key="", min_severity="high", org="", timeout=5, http_post=None
    ) -> None:
        self._url = url.rstrip("/")
        self._key = api_key
        self._min = _RANK.get(min_severity, 3)
        self._org = org
        self._timeout = timeout
        self._post = http_post or _default_post
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self.sent = 0
        self.failed = 0

    @classmethod
    def from_settings(cls) -> "TheHiveClient":
        s = settings
        return cls(
            s.thehive_url,
            s.thehive_api_key,
            s.thehive_min_severity,
            s.thehive_org,
            s.thehive_timeout_seconds,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._key)

    # ── public ───────────────────────────────────────────────
    def send_rule_alert(self, a) -> bool:
        if not self.enabled:
            return False
        hit = bool(getattr(a, "intel_hit", False))
        if _RANK.get(a.severity, 1) < self._min and not hit:
            return False
        sev = 3 if a.severity == "high" else 2 if a.severity == "medium" else 1
        if hit:
            sev = min(sev + 1, 4)
        tags = ["netguard", "signature", a.rule, a.label] + (
            ["intel-hit"] if hit else []
        )
        desc = (
            f"{a.detail}\n\nSource: {a.src_ip}\nTarget: {a.dst_ip or 'multiple'}\n"
            f"Flows: {a.flow_count}"
        )
        if getattr(a, "intel", ""):
            desc += f"\n\nThreat intel:\n{a.intel}"
        return self._send(
            {
                "type": "netguard-signature",
                "source": "netguard",
                "sourceRef": f"rule-{a.alert_id}",
                "title": f"[NetGuard] {a.rule} from {a.src_ip}",
                "description": desc,
                "severity": sev,
                "tags": tags,
                "date": int(a.timestamp * 1000),
                "tlp": 2,
                "pap": 2,
            }
        )

    def send_ml_alert(self, r) -> bool:
        if not self.enabled or not r.is_threat():
            return False
        feats = "\n".join(
            f"- {k}: SHAP {v:+.3f}"
            for k, v in sorted(
                (r.top_features or {}).items(), key=lambda kv: abs(kv[1]), reverse=True
            )
        )
        desc = (
            f"Classified {r.label} with confidence {r.confidence:.0%}.\n\n"
            f"Source: {r.src_ip}:{r.src_port}\nTarget: {r.dst_ip}:{r.dst_port}\n"
            f"Anomaly detector flagged: {bool(r.is_anomaly)}"
        )
        if feats:
            desc += f"\n\nTop features:\n{feats}"
        if getattr(r, "report", ""):
            desc += f"\n\nReport:\n{r.report}"
        return self._send(
            {
                "type": "netguard-ml",
                "source": "netguard",
                "sourceRef": f"ml-{r.flow_id}",
                "title": f"[NetGuard] {r.label} from {r.src_ip}",
                "description": desc,
                "severity": 3,
                "tags": ["netguard", "ml", r.label],
                "date": int(float(r.timestamp or time.time()) * 1000),
                "tlp": 2,
                "pap": 2,
            }
        )

    # ── internal ─────────────────────────────────────────────
    def _send(self, payload: dict) -> bool:
        ref = payload["sourceRef"]
        with self._lock:
            if ref in self._seen:
                return False
            if len(self._seen) >= _MAX_SEEN:
                self._seen.clear()
            self._seen.add(ref)
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        if self._org:
            headers["X-Organisation"] = self._org
        url = f"{self._url}/api/v1/alert"
        for attempt in (1, 2):
            try:
                code = self._post(url, payload, headers, self._timeout)
                if code in (200, 201):
                    self.sent += 1
                    return True
                if code in (400, 409):  # usually "already exists" or bad payload
                    log.warning("thehive.rejected", status=code, ref=ref)
                    self.failed += 1
                    return False
                raise RuntimeError(f"HTTP {code}")
            except Exception as exc:
                if attempt == 2:
                    self.failed += 1
                    log.warning("thehive.send_failed", error=str(exc), ref=ref)
                    with self._lock:
                        self._seen.discard(ref)  # allow a later retry
                    return False
                time.sleep(0.5)
        return False


def stats_json(c: TheHiveClient) -> str:
    return json.dumps({"enabled": c.enabled, "sent": c.sent, "failed": c.failed})
