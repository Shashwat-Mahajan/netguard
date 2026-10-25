"""
streaming/ja3.py
Matches Suricata TLS events' JA3 hashes against the abuse.ch SSLBL JA3 list.
Downloads once (cached on disk); offline falls back to the cache; never fatal.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from capture.rules import RuleAlert
from config.logger import get_logger
from config.settings import settings

log = get_logger(__name__)


def _default_fetch(url: str) -> str:
    import httpx

    r = httpx.get(url, timeout=10, follow_redirects=True)
    r.raise_for_status()
    return r.text


def parse_blocklist(text: str) -> dict[str, str]:
    """abuse.ch CSV: '# comments', then ja3_md5,first,last,listingreason."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts[0]) == 32:
            out[parts[0].lower()] = parts[3] if len(parts) > 3 else "listed"
    return out


class Ja3Blocklist:
    def __init__(
        self, url: str, cache_dir: Path | str, refresh_hours: float = 24, fetch=None
    ) -> None:
        self._url = url
        self._path = Path(cache_dir) / "ja3.csv"
        self._refresh = refresh_hours * 3600
        self._fetch = fetch or _default_fetch
        self._hashes: dict[str, str] | None = None

    @classmethod
    def from_settings(cls) -> "Ja3Blocklist":
        return cls(
            settings.ja3_url, settings.intel_dir, settings.intel_refresh_hours * 2
        )

    def _load(self) -> dict[str, str]:
        fresh = (
            self._path.exists()
            and time.time() - self._path.stat().st_mtime < self._refresh
        )
        text = ""
        if not fresh:
            try:
                text = self._fetch(self._url)
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(text, encoding="utf-8")
            except Exception as exc:
                log.warning("ja3.download_failed", error=str(exc))
        if not text and self._path.exists():
            text = self._path.read_text(encoding="utf-8", errors="replace")
        return parse_blocklist(text)

    def lookup(self, ja3: str) -> str | None:
        if self._hashes is None:
            self._hashes = self._load()
        return self._hashes.get((ja3 or "").lower())

    def size(self) -> int:
        self.lookup("")
        return len(self._hashes or {})


def ja3_alerts_from_eve(path: str | Path, bl: Ja3Blocklist) -> list[RuleAlert]:
    out: list[RuleAlert] = []
    seen: set[tuple] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event_type") != "tls":
                continue
            h = (e.get("tls", {}).get("ja3") or {}).get("hash", "")
            reason = bl.lookup(h)
            key = (e.get("src_ip"), e.get("dest_ip"), h)
            if not reason or key in seen:
                continue
            seen.add(key)
            try:
                ts = datetime.fromisoformat(e["timestamp"]).timestamp()
            except (KeyError, ValueError):
                ts = time.time()
            out.append(
                RuleAlert(
                    rule="ja3_blocklist",
                    label="Malware C2",
                    label_encoded=4,
                    severity="high",
                    src_ip=e.get("src_ip", ""),
                    dst_ip=e.get("dest_ip", ""),
                    flow_count=1,
                    detail=f"JA3 {h} listed as: {reason}",
                    timestamp=ts,
                    alert_id=str(uuid.uuid4()),
                    source="ja3",
                )
            )
    return out
