"""
ToxiLens Backend — FastAPI Application Entry Point
Loads all ML models at startup, registers API routes.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings

logger = logging.getLogger("toxilens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models into memory on startup, release on shutdown."""
    logger.info("🔬 ToxiLens starting up — loading models...")

    # Lazy imports to avoid import-time heavy loading
    from backend.app.models.ensemble import EnsemblePredictor

    app.state.predictor = EnsemblePredictor()
    app.state.predictor.load_models()

    logger.info("✅ All models loaded. ToxiLens is ready.")
    yield
    logger.info("🛑 ToxiLens shutting down.")


app = FastAPI(
    title="ToxiLens API",
    description="Interpretable Multi-Modal AI for Drug Toxicity Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
from backend.app.api.routes_predict import router as predict_router
from backend.app.api.routes_report import router as report_router
from backend.app.api.routes_derisk import router as derisk_router
from backend.app.api.routes_search import router as search_router

app.include_router(predict_router, tags=["Prediction"])
app.include_router(report_router, tags=["Report"])
app.include_router(derisk_router, tags=["De-Risking"])
app.include_router(search_router, tags=["Search"])


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ToxiLens API",
        "version": "1.0.0",
        "models_loaded": hasattr(app.state, "predictor"),
    }
