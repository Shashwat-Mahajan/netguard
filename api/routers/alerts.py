"""
api/routers/alerts.py
GET /api/v1/rule-alerts  - signature-rule alerts (ClickHouse), with threat-intel info
GET /api/v1/alerts/all   - ML + signature alerts in one timeline, tagged by source

Note: no `from __future__ import annotations` here, on purpose.
"""

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from streaming.alert_store import alert_store, merge_recent
from streaming.clickhouse_reader import ch_reader

router = APIRouter(tags=["alerts"])


def unify_alerts(ml_rows: list[dict], rule_rows: list[dict], n: int) -> list[dict]:
    """Normalise both alert kinds to one shape, newest first."""
    out = []
    for r in ml_rows:
        out.append(
            {
                "source": "ml",
                "id": r.get("flow_id", ""),
                "timestamp": r.get("timestamp") or 0,
                "rule": "classifier",
                "label": r.get("label", ""),
                "severity": "high",
                "src_ip": r.get("src_ip", ""),
                "dst_ip": r.get("dst_ip", ""),
                "detail": f"{r.get('label', '')} (confidence {r.get('confidence', 0):.2f})",
                "intel_hit": False,
            }
        )
    for r in rule_rows:
        out.append(
            {
                "source": r.get("source", "signature"),
                "id": r.get("alert_id", ""),
                "timestamp": r.get("timestamp") or 0,
                "rule": r.get("rule", ""),
                "label": r.get("label", ""),
                "severity": r.get("severity", ""),
                "src_ip": r.get("src_ip", ""),
                "dst_ip": r.get("dst_ip", ""),
                "detail": r.get("detail", ""),
                "intel_hit": bool(r.get("intel_hit")),
            }
        )
    out.sort(key=lambda a: a["timestamp"], reverse=True)
    return out[:n]


@router.get("/rule-alerts")
async def rule_alerts(
    n: int = Query(20, ge=1, le=200),
    rule: str | None = Query(None, pattern="^[a-z_]{1,32}$"),
    severity: str | None = Query(None, pattern="^(low|medium|high)$"),
):
    rows = await run_in_threadpool(ch_reader.recent_rule_alerts, n, rule, severity)
    return {"alerts": rows, "count": len(rows)}


@router.get("/alerts/all")
async def all_alerts(n: int = Query(30, ge=1, le=200)):
    ml = merge_recent(
        alert_store.recent_alerts(n),
        await run_in_threadpool(ch_reader.recent_alerts, n),
        n,
    )
    sig = await run_in_threadpool(ch_reader.recent_rule_alerts, n)
    merged = unify_alerts(ml, sig, n)
    return {"alerts": merged, "count": len(merged)}
