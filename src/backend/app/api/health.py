"""
PharmaGuard AI — Health check router.

GET /api/health  →  200 {"status": "healthy", "service": "pharmaguard-ai"}
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Returns a simple healthy response.  Used by load-balancers and CI."""
    return HealthResponse(
        status="healthy",
        service="pharmaguard-ai",
        version="0.1.0",
    )
