"""
streaming/intel.py
Threat-intel enrichment for alert IPs.

  Feodo Tracker  - botnet C2 IP list, free, no key (downloaded and cached)
  AbuseIPDB      - only if ABUSEIPDB_API_KEY is set
  AlienVault OTX - only if OTX_API_KEY is set

Private/reserved IPs are never looked up. Every failure degrades to "no data".
"""

from __future__ import annotations

import ipaddress
import json
import threading
import time
from pathlib import Path

from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)


def _default_get(url: str, headers: dict | None = None, timeout: float = 5.0):
    import httpx

    r = httpx.get(url, headers=headers or {}, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _is_public(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


class ThreatIntel:
    def __init__(
        self,
        feodo_url: str,
        cache_dir: Path | str,
        refresh_hours: float = 12,
        abuseipdb_key: str = "",
        otx_key: str = "",
        ttl_hours: float = 24,
        http_get=None,
    ) -> None:
        self._feodo_url = feodo_url
        self._dir = Path(cache_dir)
        self._refresh = refresh_hours * 3600
        self._abuse_key = abuseipdb_key
        self._otx_key = otx_key
        self._ttl = ttl_hours * 3600
        self._get = http_get or _default_get
        self._lock = threading.Lock()
        self._feodo: dict[str, dict] | None = None
        self._cache: dict[str, tuple[float, dict]] = {}

    @classmethod
    def from_settings(cls) -> "ThreatIntel":
        return cls(
            settings.feodo_url,
            settings.intel_dir,
            settings.intel_refresh_hours,
            settings.abuseipdb_api_key,
            settings.otx_api_key,
            settings.intel_cache_hours,
        )

    # ── Feodo ────────────────────────────────────────────────
    def _load_feodo(self) -> dict[str, dict]:
        path = self._dir / "feodo.json"
        data = None
        fresh = path.exists() and time.time() - path.stat().st_mtime < self._refresh
        if not fresh:
            try:
                data = self._get(self._feodo_url)
                self._dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding="utf-8")
                log.info("intel.feodo_downloaded", entries=len(data))
            except Exception as exc:
                log.warning("intel.feodo_unavailable", error=str(exc))
        if data is None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
        out: dict[str, dict] = {}
        for e in data or []:
            ip = e.get("ip_address")
            if ip:
                out[ip] = {
                    "malware": e.get("malware"),
                    "port": e.get("port"),
                    "status": e.get("status"),
                }
        return out

    def feodo_size(self) -> int:
        with self._lock:
            if self._feodo is None:
                self._feodo = self._load_feodo()
            return len(self._feodo)

    # ── Per-IP lookup ────────────────────────────────────────
    def enrich_ip(self, ip: str) -> dict:
        if not _is_public(ip):
            return {}
        now = time.time()
        with self._lock:
            hit = self._cache.get(ip)
            if hit and now - hit[0] < self._ttl:
                return hit[1]
            if self._feodo is None:
                self._feodo = self._load_feodo()
            feodo = self._feodo.get(ip)

        res: dict = {}
        if feodo:
            res["feodo"] = feodo
        if self._abuse_key:
            try:
                d = self._get(
                    f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&maxAgeInDays=90",
                    {"Key": self._abuse_key, "Accept": "application/json"},
                )["data"]
                res["abuseipdb"] = {
                    "score": d.get("abuseConfidenceScore", 0),
                    "reports": d.get("totalReports", 0),
                    "country": d.get("countryCode"),
                }
            except Exception as exc:
                log.warning("intel.abuseipdb_failed", error=str(exc))
        if self._otx_key:
            try:
                d = self._get(
                    f"https://otx.alienvault.com/api/v1/indicators/IPv4/{ip}/general",
                    {"X-OTX-API-KEY": self._otx_key},
                )
                res["otx"] = {"pulses": d.get("pulse_info", {}).get("count", 0)}
            except Exception as exc:
                log.warning("intel.otx_failed", error=str(exc))

        with self._lock:
            self._cache[ip] = (now, res)
        return res

    @staticmethod
    def is_hit(res: dict) -> bool:
        return (
            bool(res.get("feodo"))
            or res.get("abuseipdb", {}).get("score", 0) >= 50
            or res.get("otx", {}).get("pulses", 0) > 0
        )

    def enrich_alert(self, *ips: str) -> tuple[dict, bool]:
        """Returns ({ip: result} for public IPs, any_hit)."""
        info, hit = {}, False
        for ip in dict.fromkeys(i for i in ips if i):
            r = self.enrich_ip(ip)
            if r:
                info[ip] = r
                hit = hit or self.is_hit(r)
        return info, hit
