"""
Risk Scoring Engine

Computes comprehensive risk scores (0-100) from:
- Spam detection confidence
- Scam type severity
- Urgency and threat intel (suspicious URL, credential request, impersonation, etc.)
- Rule-based floor for obvious scam patterns
"""

from typing import Optional

# Scam type severity (higher = more dangerous)
SCAM_SEVERITY = {
    "bank_phishing": 0.95,
    "otp_scam": 0.90,
    "crypto_scam": 0.85,
    "loan_scam": 0.80,
    "lottery_scam": 0.75,
    "job_fraud": 0.70,
    "other": 0.60,
}

RISK_LEVELS = {
    (0, 25): "low",
    (25, 50): "medium",
    (50, 75): "high",
    (75, 100): "critical",
}

# Minimum score when multiple strong fraud indicators present (avoid false low risk)
RULE_FLOOR_THREAT_COUNT = 4
RULE_FLOOR_MIN_SPAM_CONFIDENCE = 0.35
RULE_FLOOR_MIN_SCORE = 42

RECOMMENDATIONS = {
    "low": {
        "action": "monitor",
        "description": "Monitor but no immediate action needed",
        "details": [
            "Keep the message for reference",
            "Be cautious with similar future messages",
            "No urgent action required",
        ],
    },
    "medium": {
        "action": "caution",
        "description": "Exercise caution and verify",
        "details": [
            "Do not click any links",
            "Verify sender through official channels",
            "Do not share personal information",
            "Consider reporting to spam filters",
        ],
    },
    "high": {
        "action": "block_and_report",
        "description": "Block sender and report as spam",
        "details": [
            "Immediately block the sender",
            "Report to spam/scam reporting services",
            "Do not respond under any circumstances",
            "Alert family members about similar scams",
            "Check your accounts for unauthorized activity",
        ],
    },
    "critical": {
        "action": "block_report_alert",
        "description": "Critical threat - take immediate action",
        "details": [
            "Block sender immediately",
            "Report to cybercrime authorities",
            "Check all financial accounts immediately",
            "Change passwords if any were shared",
            "Enable 2FA on all accounts",
            "Consider freezing credit if personal data was shared",
            "Alert family and friends about this scam",
        ],
    },
}


def get_risk_level(score: float) -> str:
    """Convert numeric score to risk level string."""
    for (low, high), level in RISK_LEVELS.items():
        if low <= score < high:
            return level
    return "critical" if score >= 100 else "low"


def calculate_risk_score(
    spam_confidence: float,
    scam_type: Optional[str] = None,
    urgency_score: float = 0,
    threat_indicator_count: int = 0,
    persona_vulnerability: float = 0.5,
    has_urls: bool = False,
    has_phone_numbers: bool = False,
    has_credential_request: bool = False,
    has_suspicious_url: bool = False,
    has_reward_claim: bool = False,
    has_threat: bool = False,
    has_impersonation: bool = False,
) -> dict:
    """
    Calculate comprehensive risk score (0-100) and recommended action.

    Uses all threat indicators. Applies a rule floor when threat_indicator_count
    is high and spam_confidence is non-trivial so obvious scams are not under-scored.
    """
    # Base score from spam confidence (0-38 points)
    base_score = spam_confidence * 38

    # Scam type severity (0-22 points)
    scam_severity = SCAM_SEVERITY.get(scam_type, 0.5) if scam_type else 0
    scam_score = scam_severity * 22

    # Urgency (0-8 points)
    urgency_points = min(urgency_score, 1) * 8

    # Threat indicator count (0-12 points)
    threat_score = min(threat_indicator_count / 5, 1) * 12

    # Contact / intel risk (0-20 points) — use all indicators
    contact_risk = 0
    if has_urls:
        contact_risk += 3
    if has_suspicious_url:
        contact_risk += 5
    if has_phone_numbers:
        contact_risk += 2
    if has_credential_request:
        contact_risk += 5
    if has_reward_claim:
        contact_risk += 2
    if has_threat:
        contact_risk += 3
    if has_impersonation:
        contact_risk += 2
    contact_risk = min(contact_risk, 20)

    raw_score = (
        base_score + scam_score + urgency_points + threat_score + contact_risk
    )

    # Persona modifier
    vulnerability_modifier = 1 + (persona_vulnerability - 0.5) * 0.4
    final_score = raw_score * vulnerability_modifier

    # Rule floor: many threat indicators + some spam signal → at least medium risk
    if (
        threat_indicator_count >= RULE_FLOOR_THREAT_COUNT
        and spam_confidence >= RULE_FLOOR_MIN_SPAM_CONFIDENCE
    ):
        final_score = max(final_score, RULE_FLOOR_MIN_SCORE)

    final_score = max(0, min(100, final_score))
    risk_level = get_risk_level(final_score)
    recommendation = RECOMMENDATIONS.get(risk_level, RECOMMENDATIONS["medium"])

    score_breakdown = {
        "base_spam_score": round(base_score, 2),
        "scam_severity_score": round(scam_score, 2),
        "urgency_score": round(urgency_points, 2),
        "threat_indicator_score": round(threat_score, 2),
        "contact_risk_score": round(contact_risk, 2),
        "raw_total": round(raw_score, 2),
        "vulnerability_modifier": round(vulnerability_modifier, 2),
        "final_score": round(final_score, 2),
    }

    return {
        "risk_score": round(final_score),
        "risk_level": risk_level,
        "score_breakdown": score_breakdown,
        "recommended_action": recommendation["action"],
        "action_description": recommendation["description"],
        "action_details": recommendation["details"],
        "factors": {
            "spam_detected": spam_confidence > 0.5,
            "spam_confidence": round(spam_confidence, 4),
            "scam_type": scam_type,
            "scam_severity": round(scam_severity, 2) if scam_type else None,
            "has_urls": has_urls,
            "has_phone_numbers": has_phone_numbers,
            "has_credential_request": has_credential_request,
            "has_suspicious_url": has_suspicious_url,
            "has_reward_claim": has_reward_claim,
            "has_threat": has_threat,
            "has_impersonation": has_impersonation,
            "threat_indicator_count": threat_indicator_count,
            "persona_vulnerability": round(persona_vulnerability, 2),
        },
    }


def assess_message_risk(
    analysis_result: dict,
    persona: str = "working_professional",
) -> dict:
    """Assess risk from a full message analysis result."""
    PERSONA_VULNERABILITY = {
        "elderly": 0.85,
        "student": 0.45,
        "working_professional": 0.35,
        "small_business_owner": 0.55,
    }
    vulnerability = PERSONA_VULNERABILITY.get(persona, 0.5)

    spam_confidence = analysis_result.get("spam_confidence", 0)
    scam_type = analysis_result.get("scam_type")
    features = analysis_result.get("features", {})
    urgency = features.get("urgency_normalized", 0)
    intel = analysis_result.get("threat_intelligence", {})
    threat_count = intel.get("threat_indicator_count", 0)
    indicators = intel.get("threat_indicators", {})

    return calculate_risk_score(
        spam_confidence=spam_confidence,
        scam_type=scam_type,
        urgency_score=urgency,
        threat_indicator_count=threat_count,
        persona_vulnerability=vulnerability,
        has_urls=indicators.get("has_urls", False),
        has_phone_numbers=indicators.get("has_phones", False),
        has_credential_request=indicators.get("has_credential_request", False),
        has_suspicious_url=indicators.get("has_suspicious_url", False),
        has_reward_claim=indicators.get("has_reward_claim", False),
        has_threat=indicators.get("has_threat", False),
        has_impersonation=indicators.get("has_impersonation", False),
    )
