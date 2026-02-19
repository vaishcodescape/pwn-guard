"""Risk report endpoint."""

from fastapi import APIRouter
from api.schemas import RiskReportResponse
from core.model_loader import ModelCache

router = APIRouter(tags=["Reports"])


@router.get("/risk_report", response_model=RiskReportResponse)
async def get_risk_report():
    """Get the risk report for the last analyzed message."""
    if ModelCache.last_analysis is None:
        return RiskReportResponse(
            has_analysis=False,
            analysis=None,
            message="No message analyzed yet. Use POST /analyze_message first."
        )
    return RiskReportResponse(
        has_analysis=True,
        analysis=ModelCache.last_analysis,
        message="Risk report for last analyzed message"
    )
