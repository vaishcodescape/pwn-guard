"""
Message Preprocessing Module

Handles text normalization for spam/fraud detection:
- Strips evasion tactics (zero-width chars, homoglyphs, excessive noise)
- Lowercasing with preserved ALL_CAPS signal
- URL, phone, money normalization
- Leetspeak normalization for ML while retaining leetspeak_score for features
"""

import re
import unicodedata
from typing import Optional

# Zero-width and invisible Unicode that scammers use to evade filters
ZERO_WIDTH_AND_INVISIBLE = re.compile(
    r"[\u200b-\u200d\u2060\u2061\u2062\u2063\ufeff\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e]"
)

# Common homoglyphs: Cyrillic/Greek lookalikes that can evade keyword filters
HOMOGLYPH_MAP = str.maketrans(
    "аеіоурсхун",  # Cyrillic
    "aeioprcxun",  # Latin
)

# Leetspeak: digit/symbol to letter for normalization (optional aggressive)
LEET_NORMALIZE = str.maketrans("01345$7", "oleasst")


def _strip_evasion(text: str) -> str:
    """Remove zero-width, invisible chars, and normalize homoglyphs."""
    if not text:
        return text
    cleaned = ZERO_WIDTH_AND_INVISIBLE.sub("", text)
    # Normalize Unicode to NFKC (e.g. ﬀ -> ff) to reduce variant tricks
    cleaned = unicodedata.normalize("NFKC", cleaned)
    # Replace common Cyrillic homoglyphs with Latin
    cleaned = cleaned.translate(HOMOGLYPH_MAP)
    return cleaned


def _leetspeak_ratio(text: str) -> float:
    """Ratio of chars that look like leetspeak (digits/symbols in word positions)."""
    if not text or len(text) < 3:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    digit_sym = sum(1 for c in text if c.isdigit() or c in "0$@!4")
    total = alpha + digit_sym
    return digit_sym / total if total else 0.0


def preprocess_message(text: str, metadata: Optional[dict] = None) -> dict:
    """
    Preprocess a raw text message for spam/fraud detection.

    - Strips evasion (zero-width, homoglyphs).
    - Computes capital_ratio and leetspeak_ratio from original before heavy normalization.
    - Normalizes URLs, phones, money, then remaining digits.
    - Optionally normalizes leetspeak for better TF-IDF overlap.
    """
    if not text or not text.strip():
        return {
            "original_text": text or "",
            "processed_text": "",
            "features": {
                "message_length": 0,
                "digit_count": 0,
                "url_count": 0,
                "has_urls": False,
                "capital_ratio": 0.0,
                "word_count": 0,
                "leetspeak_ratio": 0.0,
            },
            "metadata": metadata or {},
        }

    original_text = _strip_evasion(text)

    # Features from original (before lowercasing)
    url_matches = re.findall(r"https?://\S+|www\.\S+", original_text)
    url_count = len(url_matches)
    has_urls = url_count > 0
    digit_count = sum(c.isdigit() for c in original_text)
    alpha_chars = [c for c in original_text if c.isalpha()]
    capital_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )
    leet_ratio = _leetspeak_ratio(original_text)

    # Lowercase for model
    processed = original_text.lower()

    # URL normalization (tokens match training: <URL> etc. for model compatibility)
    processed = re.sub(r"https?://\S+|www\.\S+", " <URL> ", processed)

    # Phone normalization (broad international pattern)
    processed = re.sub(
        r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        " <PHONE> ",
        processed,
    )

    # Money amounts
    processed = re.sub(
        r"[\$£€₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|inr|rupees?|lakhs?|crore)",
        " <MONEY> ",
        processed,
        flags=re.IGNORECASE,
    )

    # Remaining standalone numbers
    processed = re.sub(r"\b\d+\b", " <NUM> ", processed)

    # Optional: mild leetspeak normalization so "cl1ck" and "click" match
    processed = processed.translate(LEET_NORMALIZE)

    # Whitespace and noise
    processed = re.sub(r"\s+", " ", processed).strip()
    processed = re.sub(r"[^\w\s<>!?.,]", "", processed)

    return {
        "original_text": original_text,
        "processed_text": processed,
        "features": {
            "message_length": len(original_text),
            "digit_count": digit_count,
            "url_count": url_count,
            "has_urls": has_urls,
            "capital_ratio": round(capital_ratio, 4),
            "word_count": len(original_text.split()),
            "leetspeak_ratio": round(leet_ratio, 4),
        },
        "metadata": metadata or {},
    }


def batch_preprocess(messages: list[str]) -> list[dict]:
    """Preprocess multiple messages."""
    return [preprocess_message(msg) for msg in messages]
