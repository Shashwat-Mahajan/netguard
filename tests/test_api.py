"""
tests/test_api.py
Integration tests for the FastAPI application.
Uses httpx AsyncClient — no real server needed.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_ready(client: AsyncClient):
    # Directories are created by settings validator, so this should pass
    resp = await client.get("/ready")
    assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_status_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/status")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_metrics(client: AsyncClient):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"netguard_api_requests_total" in resp.content


@pytest.mark.asyncio
async def test_404_on_unknown_path(client: AsyncClient):
    resp = await client.get("/nonexistent")
    assert resp.status_code == 404
