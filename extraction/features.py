"""
Feature Engineering Module

Extracts metadata and fraud-specific features to improve spam/scam detection:
- Urgency, action phrases, reward/claim language
- Impersonation and advance-fee signals
- Suspicious URL/TLD and "too good to be true" language
"""

import re
from typing import Optional

# Urgency / pressure (common in spam and fraud)
URGENCY_KEYWORDS = [
    "urgent", "immediately", "now", "hurry", "quick", "fast",
    "limited", "expire", "deadline", "asap", "act now", "don't wait",
    "last chance", "final notice", "warning", "alert", "important",
    "attention", "action required", "verify now", "confirm now",
    "click here", "call now", "respond now", "today only",
    "within 24 hours", "within hours", "expires today", "expiring soon",
    "immediate action", "do not ignore", "reply immediately",
]

# Action / CTA (click, call, share, pay)
ACTION_PHRASES = [
    "click here", "click the link", "click below", "tap here",
    "call this number", "call now", "call immediately",
    "send money", "transfer funds", "pay now", "payment required",
    "verify your account", "confirm your identity", "update your information",
    "enter your", "provide your", "share your", "send your",
    "otp", "pin", "password", "cvv", "card number",
    "won", "winner", "congratulations", "selected", "chosen",
    "claim your", "collect your", "redeem your",
    "register now", "sign up", "subscribe", "unsubscribe",
    "click to verify", "confirm your account", "reactivate",
]

# Impersonation: authority / institution
IMPERSONATION_KEYWORDS = [
    "bank", "irs", "tax", "police", "court", "legal",
    "support", "customer care", "security team", "official",
    "government", "ministry", "reserve bank", "rbi",
    "sbi", "hdfc", "icici", "paytm", "google", "amazon", "microsoft",
    "your account", "your card", "your bank",
]

# Advance fee / payment pressure
ADVANCE_FEE_PHRASES = [
    "processing fee", "registration fee", "transfer fee",
    "pay to unlock", "pay to receive", "pay to claim",
    "small fee", "one-time fee", "activation fee",
    "to receive", "to claim", "to unlock", "to activate",
]

# Too good to be true / reward language
TOO_GOOD_PHRASES = [
    "won", "winner", "congratulations", "selected", "prize",
    "free", "guaranteed", "100%", "double your", "triple",
    "no risk", "risk-free", "instant", "immediate",
    "pre-approved", "pre approved", "guaranteed approval",
]

# Suspicious TLDs often used in phishing/fraud
SUSPICIOUS_TLDS = (
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
    "link", "online", "site", "website", "space", "store", "tech",
    "pw", "cc", "buzz", "rest", "fit", "loan", "money", "win",
)


def _count_suspicious_tlds(text: str) -> int:
    """Count URLs containing suspicious TLDs."""
    urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text, re.IGNORECASE)
    count = 0
    for url in urls:
        url_lower = url.lower()
        for tld in SUSPICIOUS_TLDS:
            if f".{tld}" in url_lower or url_lower.endswith(f".{tld}"):
                count += 1
                break
    return count


def extract_features(text: str, preprocessed: Optional[dict] = None) -> dict:
    """Extract all relevant features from a message (raw text)."""
    text_lower = text.lower()

    message_length = len(text)
    word_count = len(text.split())
    digit_count = sum(c.isdigit() for c in text)

    urls = re.findall(r"https?://\S+|www\.\S+", text)
    url_count = len(urls)
    has_urls = url_count > 0
    suspicious_tld_count = _count_suspicious_tlds(text)

    alpha_chars = [c for c in text if c.isalpha()]
    capital_count = sum(1 for c in alpha_chars if c.isupper())
    capital_ratio = capital_count / len(alpha_chars) if alpha_chars else 0

    urgency_score = sum(1 for k in URGENCY_KEYWORDS if k in text_lower)
    urgency_normalized = min(urgency_score / 6, 1.0)

    action_phrases_found = [p for p in ACTION_PHRASES if p in text_lower]
    action_score = len(action_phrases_found)

    impersonation_score = sum(1 for k in IMPERSONATION_KEYWORDS if k in text_lower)
    impersonation_normalized = min(impersonation_score / 4, 1.0)

    advance_fee_score = sum(1 for p in ADVANCE_FEE_PHRASES if p in text_lower)
    too_good_score = sum(1 for p in TOO_GOOD_PHRASES if p in text_lower)

    exclamation_count = text.count("!")
    question_count = text.count("?")

    repetition_pattern = re.findall(r"(.)\1{2,}", text_lower)
    repetition_score = len(repetition_pattern)

    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_chars / len(text) if text else 0

    sentence_count = len(re.split(r"[.!?]+", text))
    avg_word_length = (
        sum(len(w) for w in text.split()) / word_count if word_count else 0
    )

    phone_patterns = re.findall(
        r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", text
    )
    phone_count = len(phone_patterns)

    money_patterns = re.findall(
        r"[\$£€₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:lakhs?|crore|dollars?|inr|rupees?)",
        text,
        re.IGNORECASE,
    )
    money_count = len(money_patterns)

    # Leetspeak-style: digits/symbols in place of letters (rough)
    leet_like = sum(1 for c in text if c.isdigit() or c in "0$@!")
    leetspeak_ratio = leet_like / len(text) if text else 0

    return {
        "message_length": message_length,
        "word_count": word_count,
        "digit_count": digit_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "url_count": url_count,
        "has_urls": int(has_urls),
        "suspicious_tld_count": suspicious_tld_count,
        "capital_ratio": round(capital_ratio, 4),
        "special_char_ratio": round(special_char_ratio, 4),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "urgency_score": urgency_score,
        "urgency_normalized": round(urgency_normalized, 4),
        "action_score": action_score,
        "action_phrases": action_phrases_found,
        "repetition_score": repetition_score,
        "phone_count": phone_count,
        "money_count": money_count,
        "impersonation_score": impersonation_score,
        "impersonation_normalized": round(impersonation_normalized, 4),
        "advance_fee_score": advance_fee_score,
        "too_good_score": too_good_score,
        "leetspeak_ratio": round(leetspeak_ratio, 4),
    }


def get_feature_vector(features: dict) -> list:
    """Convert feature dictionary to a numeric vector for ML models."""
    return [
        features["message_length"],
        features["word_count"],
        features["digit_count"],
        features["url_count"],
        features["has_urls"],
        features.get("suspicious_tld_count", 0),
        features["capital_ratio"],
        features["special_char_ratio"],
        features["exclamation_count"],
        features["urgency_normalized"],
        features["action_score"],
        features["repetition_score"],
        features["phone_count"],
        features["money_count"],
        features.get("impersonation_normalized", 0),
        features.get("too_good_score", 0),
    ]


FEATURE_NAMES = [
    "message_length", "word_count", "digit_count", "url_count", "has_urls",
    "suspicious_tld_count", "capital_ratio", "special_char_ratio",
    "exclamation_count", "urgency_normalized", "action_score",
    "repetition_score", "phone_count", "money_count",
    "impersonation_normalized", "too_good_score",
]
