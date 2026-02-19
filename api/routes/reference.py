"""Reference data endpoints (personas, scam types)."""

from fastapi import APIRouter
from bots.scam_baiting_bot import PERSONAS

router = APIRouter(tags=["Reference"])


@router.get("/personas")
async def list_personas():
    """Get available victim personas for chat."""
    return {
        "personas": list(PERSONAS.keys()),
        "details": {
            k: {
                "name": v.name,
                "age": v.age,
                "occupation": v.occupation,
                "background": v.background[:100] + "..."
            }
            for k, v in PERSONAS.items()
        }
    }


@router.get("/scam_types")
async def list_scam_types():
    """Get available scam types."""
    return {
        "scam_types": [
            {"type": "bank_phishing", "description": "Fake bank alerts requesting credentials"},
            {"type": "otp_scam", "description": "Tricks to steal OTP codes"},
            {"type": "job_fraud", "description": "Fake job offers requiring fees"},
            {"type": "crypto_scam", "description": "Cryptocurrency investment fraud"},
            {"type": "lottery_scam", "description": "Fake lottery/prize winning notifications"},
            {"type": "loan_scam", "description": "Fraudulent loan offers"},
            {"type": "other", "description": "Other types of scams"}
        ]
    }
