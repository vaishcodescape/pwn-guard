"""Pydantic request/response models for the API."""

from typing import Optional
from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze", min_length=1, max_length=100_000)
    sender: Optional[str] = Field(None, description="Sender identifier")
    channel: Optional[str] = Field(None, description="Channel (sms, email, etc)")

    model_config = {"json_schema_extra": {
        "example": {
            "text": "URGENT! You won $1,000,000! Click http://fakeprize.com NOW!",
            "sender": "+91-9876543210",
            "channel": "sms"
        }
    }}


class AnalysisResponse(BaseModel):
    is_spam: bool
    spam_confidence: float
    scam_type: Optional[str] = None
    scam_confidence: Optional[float] = None
    risk_score: int
    risk_level: str
    recommended_action: str
    action_details: list[str]
    threat_intelligence: dict
    features: dict
    processing_time_ms: float


class SimulateEscalationRequest(BaseModel):
    scam_type: str = Field(..., description="Scam type")
    num_turns: int = Field(5, ge=1, le=10, description="Number of escalation turns")
    initial_message: Optional[str] = Field(None, description="Seed message")

    class Config:
        json_schema_extra = {
            "example": {
                "scam_type": "bank_phishing",
                "num_turns": 5,
                "initial_message": "Your SBI account is blocked. Verify at http://sbi-fake.com"
            }
        }


class SimulateEscalationResponse(BaseModel):
    scam_type: str
    num_turns: int
    escalation: list[dict]
    message: str


class RiskReportResponse(BaseModel):
    has_analysis: bool
    analysis: Optional[dict] = None
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID")
    scammer_message: str = Field(..., description="Message from the scammer")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_001",
                "scammer_message": "Hello sir, this is SBI bank. Your account is blocked. Share OTP to unblock."
            }
        }


class ChatResponse(BaseModel):
    victim_response: str
    persona: str
    mode: str
    scam_type: str
    turn: int
    extracted_intel: dict
    tips: list[str] = Field(default_factory=list, description="Safety tips for the user")

    model_config = {"json_schema_extra": {"example": {"victim_response": "...", "persona": "Ramesh", "mode": "naive_victim", "scam_type": "otp_scam", "turn": 1, "extracted_intel": {}, "tips": ["Never share OTP"]}}}
