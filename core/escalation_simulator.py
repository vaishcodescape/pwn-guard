"""
Rule-based scam conversation escalation simulator (offline, no LLM).

Given an initial message and scam type, generates N turns of how a scam
conversation might escalate. Used for training, demos, and API responses.
"""

from __future__ import annotations

import re
from typing import Optional

# Escalation templates per scam type: list of (phase, scammer_message_template)
# Templates can use placeholders: {org}, {amount}, {url}, {phone}
ESCALATION_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "bank_phishing": [
        ("initial_contact", "URGENT: Your {org} account has been compromised. Verify immediately to avoid suspension."),
        ("urgency", "Your account will be BLOCKED in 24 hours. Click here to verify: {url}"),
        ("authority", "This is {org} security team. We need you to confirm your identity. Call {phone} now."),
        ("credential_request", "Enter your net banking username and password on the verification link. This is mandatory."),
        ("otp_request", "Share the OTP we just sent to your phone to complete verification. Reply with the 6-digit code."),
        ("threat", "Final notice: Account will be permanently closed. Legal action may follow. Verify NOW at {url}"),
        ("last_chance", "Your account is under fraud investigation. Share OTP and card details to unfreeze. Last chance."),
    ],
    "otp_scam": [
        ("initial_contact", "Your transaction of Rs.{amount} is pending. Share OTP to cancel if not you."),
        ("urgency", "OTP required within 5 minutes to stop the transaction. Reply with the code we sent."),
        ("fake_support", "This is bank verification. We sent OTP to your number. Tell us the code to secure your account."),
        ("threat", "Failure to share OTP will result in amount debited. Share code now to block transaction."),
        ("authority", "Call {phone} immediately. Our agent will guide you. Have your OTP ready."),
    ],
    "lottery_scam": [
        ("initial_contact", "Congratulations! You have won {amount} in our lottery. Claim your prize now."),
        ("urgency", "Prize expires in 48 hours. Reply with your bank details to receive payment."),
        ("fee_request", "A small processing fee of Rs.5,000 is required to release your winnings. Pay to account below."),
        ("fake_verification", "To verify winner identity, share OTP sent to your phone. Then we will transfer prize."),
        ("threat", "Legal action if you do not claim. Unclaimed prizes are reported to tax department. Pay fee now."),
    ],
    "job_fraud": [
        ("initial_contact", "You are selected for work-from-home job. Earn Rs.{amount}/month. No experience needed."),
        ("urgency", "Limited slots. Register in 24 hours at {url} to secure your position."),
        ("fee_request", "One-time registration fee of Rs.2,000 required. Pay to activate your account and start earning."),
        ("fake_training", "Complete KYC: share Aadhaar and bank details. Training material will be sent after payment."),
        ("threat", "Your slot will be given to waitlist. Pay registration now or lose this opportunity."),
    ],
    "crypto_scam": [
        ("initial_contact", "Double your crypto in 7 days. Guaranteed returns. Join our trading group. Limited seats."),
        ("urgency", "Market is moving. Invest Rs.{amount} now and get 50% bonus. Offer ends today."),
        ("fake_proof", "See our verified results: 200% returns last month. Sign up at {url} with minimum Rs.10,000."),
        ("withdrawal_trap", "To withdraw profits, pay 18% tax first. Transfer to given account. Then full amount released."),
        ("threat", "Your account is locked for verification. Pay processing fee to unlock and withdraw."),
    ],
    "loan_scam": [
        ("initial_contact", "Pre-approved loan of Rs.{amount} at 0% interest. No documents. Disbursal in 24 hours."),
        ("urgency", "Approval valid for 12 hours. Click {url} and enter details to get amount in account."),
        ("fee_request", "One-time processing fee of Rs.3,000 to be paid. After payment, loan will be credited."),
        ("fake_verification", "Share OTP and bank statement for verification. Loan amount will be released after KYC."),
        ("threat", "Your loan is approved but pending. Pay processing fee now or application will be cancelled."),
    ],
    "other": [
        ("initial_contact", "Important message regarding your account. Verify your details immediately."),
        ("urgency", "Action required within 24 hours. Click {url} or call {phone}."),
        ("generic_request", "We need to confirm your identity. Share the code we sent to your phone."),
        ("threat", "Failure to respond may result in account suspension. Contact us now."),
    ],
}


def _extract_placeholders(text: str) -> dict[str, str]:
    """Extract org, amount, url, phone from message for template filling."""
    out: dict[str, str] = {
        "org": "Bank",
        "amount": "50,000",
        "url": "https://verify-secure.com",
        "phone": "+91-9876543210",
    }
    urls = re.findall(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", text)
    if urls:
        out["url"] = urls[0].rstrip(".,;:!?)")
    phones = re.findall(
        r"\+?91[\s.-]?[6-9]\d{9}|(?<![0-9\-])[6-9]\d{9}(?![0-9\-])", text
    )
    if phones:
        out["phone"] = phones[0]
    money = re.findall(
        r"[\$£€₹]\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|inr|rupees?)",
        text,
        re.IGNORECASE,
    )
    if money:
        out["amount"] = money[0]
    for word in ["SBI", "HDFC", "ICICI", "Bank", "Paytm", "Google", "Amazon"]:
        if word.lower() in text.lower():
            out["org"] = word
            break
    return out


def simulate_escalation(
    scam_type: str,
    num_turns: int = 5,
    initial_message: Optional[str] = None,
) -> list[dict]:
    """
    Simulate scam conversation escalation (offline, rule-based).

    Args:
        scam_type: One of bank_phishing, otp_scam, lottery_scam, job_fraud, crypto_scam, loan_scam, other.
        num_turns: Number of escalation turns to generate (1–10).
        initial_message: Optional seed message; placeholders are extracted from it.

    Returns:
        List of {"turn": int, "phase": str, "scammer_message": str}.
    """
    num_turns = max(1, min(10, num_turns))
    scam_type = (scam_type or "other").lower().strip()
    if scam_type not in ESCALATION_TEMPLATES:
        scam_type = "other"

    templates = ESCALATION_TEMPLATES[scam_type]
    placeholders = _extract_placeholders(initial_message or "")

    result: list[dict] = []
    for i in range(num_turns):
        phase, template = templates[i % len(templates)]
        try:
            msg = template.format(**placeholders)
        except KeyError:
            msg = template
        result.append({
            "turn": i + 1,
            "phase": phase,
            "scammer_message": msg,
        })
    return result
