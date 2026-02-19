"""Scam baiting chat endpoints (LLM-powered)."""

from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from bots.scam_baiting_bot import ScamBaitingBot, create_bot

router = APIRouter(tags=["Scam Baiting Chat"])

chat_sessions: dict[str, ScamBaitingBot] = {}


@router.post("/chat/start")
async def start_chat_session(session_id: str = "default"):
    """Start a new scam baiting chat session."""
    chat_sessions[session_id] = create_bot()
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Chat session started. Send scam messages to /chat/respond."
    }


@router.post("/chat/respond", response_model=ChatResponse)
async def chat_respond(request: ChatRequest):
    """Send a scammer message and get the victim bot's response."""
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = create_bot()

    bot = chat_sessions[request.session_id]
    result = bot.generate_response(request.scammer_message)

    scam_type = result.get("scam_analysis", {}).get("scam_type", "unknown") if isinstance(result.get("scam_analysis"), dict) else "unknown"
    if not isinstance(scam_type, str):
        scam_type = str(scam_type) if scam_type is not None else "unknown"

    tips = []
    if scam_type == "otp_scam":
        tips = ["Never share OTP with anyone", "Banks never ask for OTP over phone"]
    elif scam_type == "bank_phishing":
        tips = ["Verify by calling official bank number", "Banks never ask for credentials over phone"]
    elif scam_type == "lottery_scam":
        tips = ["Real lotteries don't ask for fees", "If you didn't enter, you can't win"]
    else:
        tips = ["When in doubt, hang up and verify", "Never share personal info with callers"]

    return ChatResponse(
        victim_response=result.get("response") or "(Message detected as genuine — no response needed)",
        persona=result.get("persona", "Unknown"),
        mode=result.get("mode", "unknown"),
        scam_type=scam_type,
        turn=result.get("turn", 0),
        extracted_intel=result.get("extracted_intel", {}),
        tips=tips
    )


@router.get("/chat/summary/{session_id}")
async def get_chat_summary(session_id: str):
    """Get summary of a chat session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return chat_sessions[session_id].get_conversation_summary()


@router.delete("/chat/end/{session_id}")
async def end_chat_session(session_id: str):
    """End a chat session and get final summary."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    bot = chat_sessions[session_id]
    summary = bot.get_conversation_summary()
    del chat_sessions[session_id]
    return {"status": "ended", "session_id": session_id, "summary": summary}
