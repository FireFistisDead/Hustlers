"""
PharmaGuard AI — FastAPI application factory.
"""
from fastapi import FastAPI

from .api.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="PharmaGuard AI",
        description=(
            "Regulatory Intelligence Control Tower — "
            "Drug Safety Signal Detector & Regulatory Submission Readiness Checker"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(health_router, prefix="/api")

    # Future routers (added as each module is implemented):
    # app.include_router(ingest_router,   prefix="/api/ingest")
    # app.include_router(signal_router,   prefix="/api/signals")
    # app.include_router(impact_router,   prefix="/api/impact")
    # app.include_router(ctd_router,      prefix="/api/ctd")
    # app.include_router(review_router,   prefix="/api/review")
    # app.include_router(narrative_router,prefix="/api/narrative")

    return app


# ASGI entry-point used by uvicorn
app = create_app()
