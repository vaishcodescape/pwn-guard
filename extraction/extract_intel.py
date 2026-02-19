"""
Threat Intelligence Extraction Module

Extracts URLs, phone numbers, money amounts, organization names, and
fraud-specific signals (suspicious URLs, impersonation, credential pressure).
"""

import re
import threading
from urllib.parse import urlparse

try:
    import spacy
    NLP_LIB = spacy
except ImportError:
    NLP_LIB = None

NLP = None
_nlp_lock = threading.Lock()

# Trusted domains (lowercase); others can be flagged as suspicious when combined with scam language
TRUSTED_DOMAIN_KEYWORDS = (
    "google", "microsoft", "apple", "amazon", "facebook", "netflix",
    "paypal", "stripe", "gov.in", "nic.in", "indiapost", "uidai",
)
# TLDs commonly used in phishing/fraud
SUSPICIOUS_TLDS = (
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
    "link", "online", "site", "website", "space", "store", "tech",
    "pw", "cc", "buzz", "rest", "fit", "loan", "money", "win",
)
# Shorteners / redirectors
SHORTENER_DOMAINS = ("bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly")

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+")
PHONE_PATTERN = re.compile(
    r"(?:\+?91[\s.-]?[6-9]\d{9}|(?<![0-9\-])[6-9]\d{9}(?![0-9\-]))"
)
MONEY_PATTERN = re.compile(
    r"[\$£€₹]\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|inr|rupees?|pounds?|euros?|lakhs?|crore)",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Impersonation: pretending to be bank, gov, support
IMPERSONATION_PHRASES = [
    "this is your bank", "your bank", "bank security", "reserve bank",
    "customer care", "support team", "security team", "official",
    "irs", "tax department", "government", "court order",
    "sbi", "hdfc", "icici", "paytm", "google", "amazon", "microsoft",
    "verify your identity", "confirm your account", "account team",
]

ACTION_PHRASES = {
    "credential_request": [
        "enter your password", "provide your otp", "share your pin",
        "send your cvv", "confirm your card", "verify your account",
        "update your details", "enter your bank", "provide account number",
        "share otp", "send otp", "give otp", "tell us the code",
        "your login", "your password", "your pin", "your cvv",
    ],
    "urgency": [
        "act now", "immediately", "urgent", "expires today",
        "last chance", "final warning", "account suspended",
        "verify within", "confirm within", "respond immediately",
        "within 24 hours", "do not ignore",
    ],
    "reward_claim": [
        "claim your prize", "collect your winnings", "you have won",
        "congratulations", "selected as winner", "lucky winner",
        "claim reward", "redeem now", "bonus waiting",
        "you are the winner", "prize money", "claim now",
    ],
    "threat": [
        "account will be blocked", "legal action", "police complaint",
        "account suspended", "unauthorized access", "security breach",
        "immediate action required", "failure to respond",
        "account closed", "legal notice", "arrest", "lawsuit",
    ],
    "impersonation": [
        "this is", "calling from", "we are", "our team",
        "your account has been", "your card has been",
    ],
}


def get_nlp():
    """Lazy load spaCy model (thread-safe)."""
    global NLP
    if NLP is not None:
        return NLP if NLP is not False else None
    if NLP_LIB is None:
        return None
    with _nlp_lock:
        if NLP is not None:
            return NLP if NLP is not False else None
        try:
            NLP = NLP_LIB.load("en_core_web_sm")
        except OSError:
            print(
                "Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            NLP = False
    return NLP if NLP else None


def _url_risk(url: str) -> str:
    """Returns 'high', 'medium', or 'low' for a given URL."""
    url_lower = url.lower().rstrip(".,;:!?)")
    parsed = urlparse(url_lower if "://" in url_lower else "https://" + url_lower)
    netloc = parsed.netloc or parsed.path.split("/")[0]
    path = parsed.path or ""

    if any(s in netloc for s in SHORTENER_DOMAINS):
        return "high"
    try:
        parts = netloc.replace("www.", "").split(".")
        tld = parts[-1] if parts else ""
        if tld in SUSPICIOUS_TLDS:
            return "high"
    except IndexError:
        pass

    if len(netloc) < 12 and not any(t in netloc for t in TRUSTED_DOMAIN_KEYWORDS):
        return "medium"
    if any(
        w in (netloc + path)
        for w in ("verify", "secure", "account", "login", "update", "confirm")
    ):
        if not any(t in netloc for t in TRUSTED_DOMAIN_KEYWORDS):
            return "medium"

    return "low"


def extract_urls(text: str) -> list[str]:
    urls = URL_PATTERN.findall(text)
    return list(set(u.rstrip(".,;:!?)") for u in urls if u.rstrip(".,;:!?)")))


def extract_phone_numbers(text: str) -> list[str]:
    phones = PHONE_PATTERN.findall(text)
    return list(set(p for p in phones if len(re.sub(r"\D", "", p)) >= 7))


def extract_money_amounts(text: str) -> list[str]:
    return list(set(MONEY_PATTERN.findall(text)))


def extract_emails(text: str) -> list[str]:
    return list(set(EMAIL_PATTERN.findall(text)))


def extract_organizations(text: str) -> list[str]:
    nlp = get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))


def extract_action_phrases(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for category, phrases in ACTION_PHRASES.items():
        matches = [p for p in phrases if p in text_lower]
        if matches:
            found[category] = matches
    return found


def extract_threat_intelligence(text: str) -> dict:
    """
    Extract threat intelligence and fraud indicators from a message.
    Adds URL risk, impersonation, and uses all indicators for scoring.
    """
    urls = extract_urls(text)
    phones = extract_phone_numbers(text)
    money = extract_money_amounts(text)
    emails = extract_emails(text)
    orgs = extract_organizations(text)
    action_phrases = extract_action_phrases(text)

    url_risks = [_url_risk(u) for u in urls]
    high_risk_urls = sum(1 for r in url_risks if r == "high")
    has_suspicious_url = (
        high_risk_urls > 0
        or any(r == "medium" for r in url_risks)
        or (
            urls
            and not any(
                t in " ".join(urls).lower() for t in TRUSTED_DOMAIN_KEYWORDS
            )
            and any(
                s in " ".join(urls).lower()
                for s in ("bit.ly", "tinyurl", "tk", "ml", "ga")
            )
        )
    )

    has_impersonation = "impersonation" in action_phrases or any(
        p in text.lower() for p in IMPERSONATION_PHRASES
    )

    threat_indicators = {
        "has_urls": len(urls) > 0,
        "has_phones": len(phones) > 0,
        "has_money_mentions": len(money) > 0,
        "has_credential_request": "credential_request" in action_phrases,
        "has_urgency": "urgency" in action_phrases,
        "has_reward_claim": "reward_claim" in action_phrases,
        "has_threat": "threat" in action_phrases,
        "has_suspicious_url": has_suspicious_url,
        "has_impersonation": has_impersonation,
        "high_risk_url_count": high_risk_urls,
    }

    bool_count = sum(
        1 for k, v in threat_indicators.items()
        if v and k != "high_risk_url_count"
    )
    extra = min(threat_indicators.get("high_risk_url_count", 0), 2)
    threat_indicator_count = bool_count + extra

    return {
        "urls": urls,
        "url_risks": url_risks,
        "phones": phones,
        "money_amounts": money,
        "emails": emails,
        "organizations": orgs,
        "action_phrases": action_phrases,
        "threat_indicators": threat_indicators,
        "threat_indicator_count": threat_indicator_count,
    }
