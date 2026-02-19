"""
Message analysis endpoint.

Runs spam/scam models and threat intel, then risk scoring.
Applies a rule-based override when fraud indicators are strong but the model says ham.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException

from api.schemas import MessageRequest, AnalysisResponse
from core.model_loader import ModelCache
from core.risk_engine import calculate_risk_score
from extraction.preprocess import preprocess_message
from extraction.features import extract_features
from extraction.extract_intel import extract_threat_intelligence

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analyze")

router = APIRouter(tags=["Analysis"])

SPAM_THRESHOLD = 0.45
RULE_OVERRIDE_INDICATORS = ("has_credential_request", "has_suspicious_url", "has_urgency")
RULE_OVERRIDE_MIN_INDICATORS = 2
RULE_OVERRIDE_THREAT_COUNT = 4
RULE_OVERRIDE_SPAM_CONFIDENCE = 0.68


def _should_override_to_spam(
    is_spam: bool,
    spam_proba: float,
    threat_intel: dict,
) -> tuple[bool, float]:
    if is_spam:
        return is_spam, spam_proba
    indicators = threat_intel.get("threat_indicators", {})
    count_present = sum(1 for k in RULE_OVERRIDE_INDICATORS if indicators.get(k))
    threat_count = threat_intel.get("threat_indicator_count", 0)
    if count_present >= RULE_OVERRIDE_MIN_INDICATORS or threat_count >= RULE_OVERRIDE_THREAT_COUNT:
        return True, max(spam_proba, RULE_OVERRIDE_SPAM_CONFIDENCE)
    return is_spam, spam_proba


@router.post("/analyze_message", response_model=AnalysisResponse)
async def analyze_message(request: MessageRequest):
    start_time = time.time()

    if ModelCache.spam_model is None:
        raise HTTPException(
            status_code=503,
            detail="Spam detection model not loaded. Run training first: python -m training.train_spam",
        )

    text = request.text
    metadata = {"sender": request.sender, "channel": request.channel}

    preprocessed = preprocess_message(text, metadata)
    # Run feature and intel extraction in parallel (both CPU-bound, independent)
    future_features = _EXECUTOR.submit(extract_features, text)
    future_intel = _EXECUTOR.submit(extract_threat_intelligence, text)
    features = future_features.result()
    threat_intel = future_intel.result()

    X_spam = ModelCache.spam_vectorizer.transform([preprocessed["processed_text"]])
    spam_proba_arr = ModelCache.spam_model.predict_proba(X_spam)[0]
    spam_proba = float(spam_proba_arr[1])

    is_spam = spam_proba >= SPAM_THRESHOLD

    is_spam, spam_confidence = _should_override_to_spam(
        is_spam, spam_proba, threat_intel
    )
    if is_spam and spam_confidence != spam_proba:
        spam_proba = spam_confidence

    scam_type = None
    scam_confidence = None
    if is_spam and ModelCache.scam_model is not None:
        X_scam = ModelCache.scam_vectorizer.transform([preprocessed["processed_text"]])
        raw_scam_type = ModelCache.scam_model.predict(X_scam)[0]
        scam_type = str(raw_scam_type) if raw_scam_type is not None else None
        scam_proba_arr = ModelCache.scam_model.predict_proba(X_scam)[0]
        scam_confidence = float(max(scam_proba_arr))

    ind = threat_intel.get("threat_indicators", {})

    risk_result = calculate_risk_score(
        spam_confidence=spam_proba,
        scam_type=scam_type,
        urgency_score=features.get("urgency_normalized", 0),
        threat_indicator_count=threat_intel.get("threat_indicator_count", 0),
        persona_vulnerability=0.5,
        has_urls=ind.get("has_urls", False),
        has_phone_numbers=ind.get("has_phones", False),
        has_credential_request=ind.get("has_credential_request", False),
        has_suspicious_url=ind.get("has_suspicious_url", False),
        has_reward_claim=ind.get("has_reward_claim", False),
        has_threat=ind.get("has_threat", False),
        has_impersonation=ind.get("has_impersonation", False),
    )

    processing_time = (time.time() - start_time) * 1000

    ModelCache.last_analysis = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "is_spam": is_spam,
        "spam_confidence": spam_confidence,
        "scam_type": scam_type,
        "scam_confidence": scam_confidence,
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "recommended_action": risk_result["recommended_action"],
        "action_details": risk_result["action_details"],
        "threat_intelligence": threat_intel,
        "features": features,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return AnalysisResponse(
        is_spam=is_spam,
        spam_confidence=round(spam_confidence, 4),
        scam_type=scam_type,
        scam_confidence=round(scam_confidence, 4) if scam_confidence else None,
        risk_score=risk_result["risk_score"],
        risk_level=risk_result["risk_level"],
        recommended_action=risk_result["recommended_action"],
        action_details=risk_result.get("action_details", []),
        threat_intelligence=threat_intel,
        features=features,
        processing_time_ms=round(processing_time, 2),
    )
