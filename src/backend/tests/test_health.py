"""
Tests for GET /api/health
"""
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import app


@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_body():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/health")

    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "pharmaguard-ai"
    assert "version" in body
