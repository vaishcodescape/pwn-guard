"""Risk report endpoint."""

from fastapi import APIRouter
from api.schemas import RiskReportResponse
from core.model_loader import ModelCache

router = APIRouter(tags=["Reports"])


def _ensure_json_serializable(obj):
    """Recursively coerce numpy/other non-JSON types to native Python for safe JSON response."""
    if obj is None:
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(v) for v in obj]
    return obj


@router.get("/risk_report", response_model=RiskReportResponse)
async def get_risk_report():
    """Get the risk report for the last analyzed message."""
    if ModelCache.last_analysis is None:
        return RiskReportResponse(
            has_analysis=False,
            analysis=None,
            message="No message analyzed yet. Use POST /analyze_message first."
        )
    analysis = _ensure_json_serializable(ModelCache.last_analysis)
    return RiskReportResponse(
        has_analysis=True,
        analysis=analysis,
        message="Risk report for last analyzed message"
    )
