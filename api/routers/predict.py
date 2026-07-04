"""
api/routers/predict.py
Prediction endpoints — Phase 3.

POST /api/v1/predict   — classify a single flow
GET  /api/v1/flows     — recent flows from alert store
GET  /api/v1/alerts    — recent threats only
GET  /api/v1/stats     — live dashboard stats
WS   /api/v1/stream    — WebSocket live alert feed
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from config.logger import get_logger
from processor.feature_schema import FlowFeatures
from streaming.alert_store import alert_store
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
    src_ip:               str   = Field(..., examples=["192.168.1.10"])
    dst_ip:               str   = Field(..., examples=["8.8.8.8"])
    src_port:             int   = Field(..., ge=0, le=65535)
    dst_port:             int   = Field(..., ge=0, le=65535)
    protocol:             int   = Field(default=6)
    duration:             float = Field(default=0.0)
    fwd_packet_count:     int   = Field(default=0)
    bwd_packet_count:     int   = Field(default=0)
    total_fwd_bytes:      int   = Field(default=0)
    total_bwd_bytes:      int   = Field(default=0)
    flow_bytes_per_sec:   float = Field(default=0.0)
    flow_packets_per_sec: float = Field(default=0.0)
    syn_flag_count:       int   = Field(default=0)
    fin_flag_count:       int   = Field(default=0)
    rst_flag_count:       int   = Field(default=0)
    psh_flag_count:       int   = Field(default=0)


class PredictionResponse(BaseModel):
    flow_id:       str
    label:         str
    label_encoded: int
    confidence:    float
    is_anomaly:    bool
    anomaly_score: float
    class_probs:   dict[str, float]
    is_threat:     bool
    inference_ms:  float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def predict_flow(req: FlowRequest):
    """Classify a single network flow and return attack type + confidence."""
    engine = get_engine()

    flow = FlowFeatures(
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

    result = engine.predict(flow)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run Phase 2 training first.",
        )

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
        inference_ms=result.inference_ms,
    )


@router.get("/flows")
async def get_recent_flows(n: int = 50):
    """Return n most recent classified flows."""
    return {"flows": alert_store.recent_flows(min(n, 200)), "count": n}


@router.get("/alerts")
async def get_recent_alerts(n: int = 20):
    """Return n most recent threat alerts."""
    return {"alerts": alert_store.recent_alerts(min(n, 100)), "count": n}


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
    WebSocket live feed — pushes alerts and stats every 2 seconds.
    React dashboard connects here for real-time updates.
    """
    await ws.accept()
    log.info("ws.connected", client=str(ws.client))
    last_count = 0

    try:
        while True:
            alerts = alert_store.recent_alerts(5)
            if len(alerts) != last_count:
                last_count = len(alerts)
                if alerts:
                    await ws.send_text(json.dumps({
                        "type": "alert",
                        "data": alerts[0],
                    }))

            await ws.send_text(json.dumps({
                "type":  "stats",
                "data":  alert_store.summary_stats(),
            }))
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        log.info("ws.disconnected")
    except Exception as exc:
        log.warning("ws.error", error=str(exc))
