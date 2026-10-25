"""
streaming/suricata.py
Parses Suricata EVE JSON alert events into NetGuard alerts (source='suricata').
Missing or malformed lines are skipped, never fatal.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from capture.rules import RuleAlert

_SEV = {1: "high", 2: "medium", 3: "low"}


def _label(category: str, msg: str) -> tuple[str, int]:
    t = f"{category} {msg}".lower()
    if "dos" in t or "denial" in t or "flood" in t:
        return "DoS", 1
    if "scan" in t or "recon" in t or "probe" in t:
        return "Probe", 2
    if "brute" in t or "login" in t or "credential" in t:
        return "R2L", 3
    return "Unknown", -1


def parse_eve_line(line: str) -> RuleAlert | None:
    try:
        e = json.loads(line)
    except ValueError:
        return None
    if e.get("event_type") != "alert":
        return None
    a = e.get("alert", {})
    try:
        ts = datetime.fromisoformat(e["timestamp"]).timestamp()
    except (KeyError, ValueError):
        return None
    label, enc = _label(a.get("category", ""), a.get("signature", ""))
    detail = f"{a.get('signature', '')} (sid {a.get('signature_id', '?')})"
    return RuleAlert(
        rule=f"suricata:{a.get('signature_id', 'unknown')}",
        label=label,
        label_encoded=enc,
        severity=_SEV.get(a.get("severity"), "medium"),
        src_ip=e.get("src_ip", ""),
        dst_ip=e.get("dest_ip", ""),
        flow_count=1,
        detail=detail,
        timestamp=ts,
        alert_id=str(uuid.uuid4()),
        source="suricata",
    )


def parse_eve_file(path: str | Path) -> list[RuleAlert]:
    out: list[RuleAlert] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            a = parse_eve_line(line)
            if a:
                out.append(a)
    return out
