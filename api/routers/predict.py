"""
api/routers/predict.py
Prediction endpoints.

POST /api/v1/predict   — classify a flow + SHAP explanation + incident report (admin, rate limited)
GET  /api/v1/flows     — recent flows (ClickHouse + in-memory cache)
GET  /api/v1/alerts    — recent threats only (ClickHouse + in-memory cache)
GET  /api/v1/stats     — live dashboard stats
WS   /api/v1/stream    — WebSocket live alert feed

Note: no `from __future__ import annotations` here, on purpose (slowapi).
"""

import asyncio
import json
import time
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.auth import require_admin
from api.limiter import limiter
from config.logger import get_logger
from config.settings import settings
from models.incident_reporter import get_reporter
from processor.feature_schema import FlowFeatures
from streaming.alert_store import alert_store, merge_recent
from streaming.clickhouse_reader import ch_reader
from streaming.clickhouse_writer import ch_writer

log = get_logger(__name__)
router = APIRouter(tags=["predict"])

# ── Lazy model loading ────────────────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from streaming.consumer import InferenceEngine

        _engine = InferenceEngine()
        _engine.load()
    return _engine


# ── Schemas ───────────────────────────────────────────────────────────────────


class FlowRequest(BaseModel):
    src_ip: str = Field(..., examples=["192.168.1.10"])
    dst_ip: str = Field(..., examples=["8.8.8.8"])
    src_port: int = Field(..., ge=0, le=65535)
    dst_port: int = Field(..., ge=0, le=65535)
    protocol: int = Field(default=6)
    duration: float = Field(default=0.0)
    fwd_packet_count: int = Field(default=0)
    bwd_packet_count: int = Field(default=0)
    total_fwd_bytes: int = Field(default=0)
    total_bwd_bytes: int = Field(default=0)
    flow_bytes_per_sec: float = Field(default=0.0)
    flow_packets_per_sec: float = Field(default=0.0)
    syn_flag_count: int = Field(default=0)
    fin_flag_count: int = Field(default=0)
    rst_flag_count: int = Field(default=0)
    psh_flag_count: int = Field(default=0)


class PredictionResponse(BaseModel):
    flow_id: str
    label: str
    label_encoded: int
    confidence: float
    is_anomaly: bool
    anomaly_score: float
    class_probs: dict[str, float]
    is_threat: bool
    severity: str = "none"
    inference_ms: float
    top_features: dict[str, float] = {}
    report: str | None = None
    report_backend: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/predict", response_model=PredictionResponse, dependencies=[Depends(require_admin)]
)
@limiter.limit(lambda: settings.rate_limit_predict)
async def predict_flow(
    request: Request,
    req: FlowRequest,
    report: bool | None = Query(
        None, description="Generate an incident report. Default: only for alerts."
    ),
):
    """Classify a flow: attack type, SHAP top features, and a plain-English report."""
    engine = get_engine()

    flow = FlowFeatures(
        flow_id=str(uuid.uuid4()),
        src_ip=req.src_ip,
        dst_ip=req.dst_ip,
        src_port=req.src_port,
        dst_port=req.dst_port,
        protocol=req.protocol,
        duration=req.duration,
        fwd_packet_count=req.fwd_packet_count,
        bwd_packet_count=req.bwd_packet_count,
        total_fwd_bytes=req.total_fwd_bytes,
        total_bwd_bytes=req.total_bwd_bytes,
        flow_bytes_per_sec=req.flow_bytes_per_sec,
        flow_packets_per_sec=req.flow_packets_per_sec,
        syn_flag_count=req.syn_flag_count,
        fin_flag_count=req.fin_flag_count,
        rst_flag_count=req.rst_flag_count,
        psh_flag_count=req.psh_flag_count,
    )

    result = await run_in_threadpool(engine.predict, flow, True)  # always explain
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run Phase 2 training first.",
        )

    want_report = (
        report if report is not None else (result.is_threat() or result.is_suspicious())
    )
    if want_report:
        rep = await run_in_threadpool(
            get_reporter().generate,
            result.to_dict(),
            result.label,
            result.confidence,
            result.top_features,
        )
        result.report = rep["report"]
        result.report_backend = rep["backend"]

    alert_store.add(result)
    ch_writer.write(result)

    return PredictionResponse(
        flow_id=result.flow_id,
        label=result.label,
        label_encoded=result.label_encoded,
        confidence=result.confidence,
        is_anomaly=result.is_anomaly,
        anomaly_score=result.anomaly_score,
        class_probs=result.class_probs,
        is_threat=result.is_threat(),
        severity=result.severity(),
        inference_ms=result.inference_ms,
        top_features=result.top_features,
        report=result.report or None,
        report_backend=result.report_backend or None,
    )


@router.get("/flows")
async def get_recent_flows(n: int = 50):
    """Return n most recent classified flows (persisted + not-yet-flushed)."""
    n = max(1, min(n, 200))
    cached = alert_store.recent_flows(n)
    stored = await run_in_threadpool(ch_reader.recent_flows, n)
    flows = merge_recent(cached, stored, n)
    return {"flows": flows, "count": len(flows)}


@router.get("/alerts")
async def get_recent_alerts(n: int = 20):
    """Return n most recent threat alerts (persisted + not-yet-flushed)."""
    n = max(1, min(n, 100))
    cached = alert_store.recent_alerts(n)
    stored = await run_in_threadpool(ch_reader.recent_alerts, n)
    alerts = merge_recent(cached, stored, n)
    return {"alerts": alerts, "count": len(alerts)}


@router.get("/stats")
async def get_stats():
    """Live dashboard statistics — total flows, threat rate, label counts."""
    engine = get_engine()
    return {
        **alert_store.summary_stats(),
        "inference_engine": engine.stats,
        "timestamp": time.time(),
    }


@router.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    """
    WebSocket live feed — pushes new alerts immediately and stats every 2 seconds.
    React dashboard connects here for real-time updates.
    """
    await ws.accept()
    log.info("ws.connected", client=str(ws.client))
    last_alert_id: str | None = None

    try:
        while True:
            newest = alert_store.recent_alerts(1)
            if newest and newest[0].get("flow_id") != last_alert_id:
                last_alert_id = newest[0].get("flow_id")
                await ws.send_text(
                    json.dumps({"type": "alert", "data": newest[0]}, default=str)
                )

            await ws.send_text(
                json.dumps(
                    {"type": "stats", "data": alert_store.summary_stats()}, default=str
                )
            )
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        log.info("ws.disconnected")
    except Exception as exc:
        log.warning("ws.error", error=str(exc))
