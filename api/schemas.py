"""
api/schemas.py
Pydantic v2 models for all API request/response bodies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: float


class FlowRecord(BaseModel):
    """A single network flow record submitted for prediction (Phase 3)."""

    src_ip: str = Field(..., examples=["192.168.1.10"])
    dst_ip: str = Field(..., examples=["8.8.8.8"])
    src_port: int = Field(..., ge=0, le=65535)
    dst_port: int = Field(..., ge=0, le=65535)
    protocol: int = Field(..., ge=0, le=255)
    duration: float = Field(..., ge=0)
    fwd_packet_count: int = Field(default=0, ge=0)
    bwd_packet_count: int = Field(default=0, ge=0)
    total_fwd_bytes: int = Field(default=0, ge=0)
    total_bwd_bytes: int = Field(default=0, ge=0)


class PredictionResponse(BaseModel):
    """ML prediction result (Phase 2/3)."""

    flow_id: str
    label: str
    label_encoded: int
    confidence: float
    top_features: dict[str, float]  # SHAP values
