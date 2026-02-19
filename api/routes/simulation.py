"""Rule-based escalation simulation endpoint."""

from fastapi import APIRouter
from api.schemas import SimulateEscalationRequest, SimulateEscalationResponse
from core.escalation_simulator import simulate_escalation

router = APIRouter(tags=["Simulation"])


@router.post("/simulate_escalation", response_model=SimulateEscalationResponse)
async def simulate_escalation_endpoint(request: SimulateEscalationRequest):
    """Simulate scam conversation escalation (offline, rule-based). No LLM required."""
    escalation = simulate_escalation(
        scam_type=request.scam_type,
        num_turns=request.num_turns,
        initial_message=request.initial_message,
    )
    return SimulateEscalationResponse(
        scam_type=request.scam_type,
        num_turns=len(escalation),
        escalation=escalation,
        message="Rule-based escalation simulation (offline)."
    )
