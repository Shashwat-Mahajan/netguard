"""
streaming/zeek.py
Turns Zeek notice.log entries into NetGuard alerts (source='zeek').
Missing files and odd entries are skipped, never fatal.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path

from capture.rules import RuleAlert
from capture.zeek_parser import ZeekNoticeParser

_SEVERITIES = ("high", "medium", "low")


def _label(note: str) -> tuple[str, int]:
    n = note.lower()
    if "brute" in n:
        return "R2L", 3
    if "scan" in n:
        return "Probe", 2
    return "Unknown", -1


def _epoch(ts) -> float:
    if isinstance(ts, datetime):
        return ts.timestamp()
    try:
        return float(ts)
    except (TypeError, ValueError):
        return time.time()


def notice_to_alert(n) -> RuleAlert:
    note = str(getattr(n, "note", "") or "unknown")
    msg = str(getattr(n, "msg", "") or "")
    sub = str(getattr(n, "sub", "") or "")
    label, enc = _label(note)
    severity = getattr(n, "severity", "low")
    if severity not in _SEVERITIES:
        severity = "low"
    detail = f"Zeek {note}"
    if msg:
        detail += f": {msg}"
    if sub:
        detail += f" ({sub})"
    return RuleAlert(
        rule=f"zeek:{note}",
        label=label,
        label_encoded=enc,
        severity=severity,
        src_ip=str(getattr(n, "src_ip", "") or ""),
        dst_ip=str(getattr(n, "dst_ip", "") or ""),
        flow_count=1,
        detail=detail,
        timestamp=_epoch(getattr(n, "timestamp", None)),
        alert_id=str(uuid.uuid4()),
        source="zeek",
    )


def parse_notice_file(path: str | Path) -> list[RuleAlert]:
    return [notice_to_alert(n) for n in ZeekNoticeParser().parse_file(path)]


def ingest_notices(path: str | Path, writer, restamp: bool = False) -> tuple[int, int]:
    """Write every notice in `path` through `writer` (already started).
    Returns (written, found)."""
    alerts = parse_notice_file(path)
    if restamp:
        now = time.time()
        for a in alerts:
            a.timestamp = now
    before = writer.written
    for a in alerts:
        writer.write(a)
    return writer.written - before, len(alerts)
