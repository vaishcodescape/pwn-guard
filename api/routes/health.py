"""Health and readiness endpoints."""

from fastapi import APIRouter
from config import API_VERSION
from core.model_loader import ModelCache

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """Root: service info and model status."""
    return {
        "status": "healthy",
        "service": "Spam Detection & Scam Simulation API",
        "version": API_VERSION,
        "models_loaded": {
            "spam_detector": ModelCache.spam_model is not None,
            "scam_classifier": ModelCache.scam_model is not None
        }
    }


@router.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "version": API_VERSION}


@router.get("/ready")
async def ready():
    """Readiness probe: true when models are loaded."""
    return {
        "ready": ModelCache.spam_model is not None,
        "spam_detector": ModelCache.spam_model is not None,
        "scam_classifier": ModelCache.scam_model is not None
    }
