"""
Scam Baiting Bot - LLM-Powered Anti-Scam Conversational AI

An intelligent bot that:
1. Analyzes scam messages to understand context
2. Uses LLM to generate contextually appropriate victim responses
3. Selects persona based on message analysis
4. Keeps scammers busy while extracting useful intel
5. Adapts responses based on conversation flow

Uses Ollama for local LLM inference (CPU-friendly).
"""

import os
import re
import json
import requests
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from extraction.preprocess import preprocess_message
from extraction.features import extract_features
from extraction.extract_intel import extract_threat_intelligence
from config import MODEL_DIR

# Try to load spam detection models
try:
    import joblib
    SPAM_DETECTOR = joblib.load(os.path.join(MODEL_DIR, "spam_detector.joblib"))
    SPAM_VECTORIZER = joblib.load(os.path.join(MODEL_DIR, "spam_vectorizer.joblib"))
    SPAM_DETECTION_AVAILABLE = True
except Exception as e:
    SPAM_DETECTOR = None
    SPAM_VECTORIZER = None
    SPAM_DETECTION_AVAILABLE = False
    print(f"⚠ Spam detection models not loaded: {e}")




def is_likely_genuine(message: str, metadata: dict = None) -> bool:
    """Check if a message is likely genuine using heuristics, regex patterns, and metadata.
    
    This acts as a whitelist to fast-track safe transactional messages (OTPs, codes, banking alerts)
    that often get flagged as spam by the model.
    """
    if metadata:
        # Trust explicit safe flags from upstream systems
        if metadata.get('is_safe', False) or metadata.get('risk_score', 100) < 10:
            return True
            
        # Trust specific senders if whitelist exists (placeholder)
        sender = metadata.get('sender', '')
        if sender and sender.lower() in ['bank', 'verify', 'auth']: # simplistic example
            return True

    message_lower = message.lower()
    
    # 1. Regex Patterns for Genuine Transactional Structures
    transactional_patterns = [
        # Debited/Credited with Amount
        r'(?:acct|a/c|account)[\s\w*]*\d+\s+(?:debited|credited)\s+(?:for|with)?\s*(?:rs\.?|inr|\$)\s*\d+',
        r'(?:rs\.?|inr|\$)\s*\d+\s+(?:debited|credited)\s+(?:from|to)\s+(?:acct|a/c|account)',
        
        # Balance Info
        r'(?:avl|available|cl|clear|eff)\s+bal(?:ance)?\s*[:\-\s]\s*(?:rs\.?|inr|\$)?\s*\d+',
        
        # OTP / Verification Codes (Specific & Structured)
        r'(?:otp|code)\s+(?:is|for)\s+[:\-\s]*\w{4,8}',
        r'\w{4,8}\s+is\s+your\s+(?:otp|verification)',
        
        # Reference Numbers
        r'(?:ref|txn)\s*(?:no|id)?\s*[:\-\s]\s*[\w\d]+',
    ]
    
    is_transactional = any(re.search(p, message_lower) for p in transactional_patterns)

    # 2. Safe Keywords (fallback if regex misses but still looks like OTP)
    safe_keywords = ['code is', 'verification code', 'otp is', 'one time password', 'secret code']
    has_safe_keyword = any(kw in message_lower for kw in safe_keywords)
    
    # 3. Unsafe Indicators - Refined
    # We allow "Call" if it is followed by 1800 (Toll-Free)
    if 'call' in message_lower:
        # Check if it is a 1800 number
        if re.search(r'call\s*(?:\+91)?\s*1800', message_lower):
            pass # Safe call
        else:
             # Unsafe call (could be scammer mobile)
             return False

    unsafe_keywords = [
        'urgent', 'immediate', 'click', 'link', 'http', 'www', 'won', 'prize', 
        'lottery', 'blocked', 'suspended', 'expire', 'act now'
    ]
    
    has_unsafe_keyword = any(kw in message_lower for kw in unsafe_keywords)
    
    # Whitelist Logic
    
    # Case A: Valid Transactional Pattern found + No unsafe keywords
    if is_transactional and not has_unsafe_keyword:
        return True
        
    # Case B: Traditional OTP format (short + safe keyword) + No unsafe keywords
    if has_safe_keyword and not has_unsafe_keyword:
        if len(message.split()) < 30: # Increased word limit slightly for structured alerts
            return True
            
    return False


def is_spam_message(message: str, threshold: float = 0.5, metadata: dict = None) -> tuple[bool, float]:
    """Check if a message is spam using the trained classifier.
    
    Args:
        message: The message to check
        threshold: Probability threshold for spam classification (default 0.5)
    
    Returns:
        Tuple of (is_spam: bool, confidence: float)
    """
    # 1. Whitelist Check (Fast Pass)
    if is_likely_genuine(message, metadata):
        return False, 0.0

    # 2. Heuristic/Model Check
    if not SPAM_DETECTION_AVAILABLE:
        # Fallback: use keyword-based detection
        spam_keywords = [
            'urgent', 'blocked', 'suspended', 'verify', 'otp', 'kyc', 
            'won', 'lottery', 'prize', 'click', 'link', 'account',
            'earn', 'job', 'work from home', 'investment', 'guaranteed',
            'congratulations', 'selected', 'winner', 'claim', 'reward'
        ]
        message_lower = message.lower()
        matches = sum(1 for kw in spam_keywords if kw in message_lower)
        is_spam = matches >= 2  # At least 2 spam keywords
        confidence = min(matches / 5, 1.0)  # Normalize confidence
        return is_spam, confidence
    
    try:
        # Use the trained model
        preprocessed = preprocess_message(message)
        vectorized = SPAM_VECTORIZER.transform([preprocessed['processed_text']])
        
        # Get probability if available
        if hasattr(SPAM_DETECTOR, 'predict_proba'):
            proba = SPAM_DETECTOR.predict_proba(vectorized)[0]
            spam_confidence = proba[1] if len(proba) > 1 else proba[0]
            is_spam = spam_confidence >= threshold
            return is_spam, float(spam_confidence)
        else:
            prediction = SPAM_DETECTOR.predict(vectorized)[0]
            is_spam = prediction == 1 or prediction == 'spam'
            return is_spam, 1.0 if is_spam else 0.0
            
    except Exception as e:
        print(f"Spam detection error: {e}")
        return True, 0.5  # Default to treating as potential spam


# ===================== Configuration =====================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "sarah:latest"  # Change to your model (try llama3.2:1b for speed)
OLLAMA_TIMEOUT = 60  # Reduced timeout
OLLAMA_MAX_RETRIES = 2  # Reduced retries for speed
USE_STREAMING = True  # Stream responses for faster perceived performance
USE_FAST_MODE = True  # Use fallback for first 1-2 turns
SPAM_THRESHOLD = 0.5  # Threshold for spam detection


# ===================== Enums & Data Classes =====================

class BotMode(Enum):
    NAIVE_VICTIM = "naive_victim"
    TIME_WASTING = "time_wasting" 
    INFO_EXTRACTION = "info_extraction"
    CONFUSED = "confused"


class ScamType(Enum):
    BANK_PHISHING = "bank_phishing"
    OTP_SCAM = "otp_scam"
    LOTTERY = "lottery_scam"
    JOB_FRAUD = "job_fraud"
    CRYPTO = "crypto_scam"
    LOAN = "loan_scam"
    ROMANCE = "romance_scam"
    TECH_SUPPORT = "tech_support_scam"
    UNKNOWN = "unknown"


@dataclass
class Persona:
    name: str
    age: int
    occupation: str
    background: str
    speaking_style: str
    vulnerabilities: list
    knowledge_gaps: list  # Things this persona doesn't understand


# ===================== Persona Definitions =====================

PERSONAS = {
    'elderly_retiree': Persona(
        name="Ramesh Sharma",
        age=68,
        occupation="Retired government employee",
        background="Lives alone, wife passed away, children are abroad. Gets pension deposited monthly. Not tech-savvy, uses basic smartphone. Trusts authority figures.",
        speaking_style="Speaks slowly, uses 'beta', 'haan ji', asks same question twice, mentions grandchildren, says things like 'my son usually helps me with this'",
        vulnerabilities=["trusts authority", "worried about money", "lonely", "slow with technology"],
        knowledge_gaps=["OTP security", "online fraud", "phishing links", "UPI", "doesn't know banks don't call"]
    ),
    'college_student': Persona(
        name="Priya Patel", 
        age=21,
        occupation="Final year engineering student",
        background="Looking for job/internship, limited savings, lives in hostel. Has UPI apps, somewhat tech-aware but naive about financial scams. Excited about money.",
        speaking_style="Uses short sentences, says 'omg', 'wait what', 'seriously?', asks 'is this real?', gets excited quickly",
        vulnerabilities=["needs money for fees", "excited about job offers", "trusts official-sounding calls", "FOMO"],
        knowledge_gaps=["job scam tactics", "too-good-to-be-true offers", "advance fee frauds"]
    ),
    'busy_professional': Persona(
        name="Vikram Mehta",
        age=38,
        occupation="IT Manager at MNC",
        background="Busy schedule, handles multiple bank accounts, travels frequently. More aware but distracted. Might fall for sophisticated scams if caught off-guard.",
        speaking_style="Quick responses, says 'make it fast', 'I'm in a meeting', 'which account?', multi-tasks while talking",
        vulnerabilities=["distracted", "handles multiple accounts", "might rush decisions", "ego - thinks can't be scammed"],
        knowledge_gaps=["newest scam tactics", "social engineering", "may let guard down when tired"]
    ),
    'small_business_owner': Persona(
        name="Sunita Agarwal",
        age=52,
        occupation="Owns small textile shop",
        background="Has business and personal accounts, deals with many vendors, uses WhatsApp for business. Gets legitimate urgent calls, harder to distinguish scams.",
        speaking_style="Business-like, asks for details, mentions 'let me check with accountant', 'which payment', talks about shop",
        vulnerabilities=["many real urgent calls", "complex finances", "trusts vendor relationships"],
        knowledge_gaps=["digital payment frauds", "impersonation scams", "business email compromise"]
    )
}


def select_persona_for_scam(scam_type: str) -> Persona:
    """Select the most appropriate persona based on scam type.
    
    Mapping logic:
    - Job/work from home scams → College student (eager job seeker)
    - Crypto/investment scams → IT professional (has investment knowledge)
    - Loan/business scams → Small business owner (needs capital)
    - Lottery/bank/OTP scams → Elderly retiree (trusts authority, easily confused)
    """
    scam_type_lower = scam_type.lower() if scam_type else 'unknown'
    
    persona_map = {
        'job_fraud': 'college_student',
        'job_scam': 'college_student',
        'work_from_home': 'college_student',
        'crypto_scam': 'busy_professional',
        'investment_scam': 'busy_professional',
        'trading_scam': 'busy_professional',
        'loan_scam': 'small_business_owner',
        'business_scam': 'small_business_owner',
        'lottery_scam': 'elderly_retiree',
        'lottery': 'elderly_retiree',
        'bank_phishing': 'elderly_retiree',
        'otp_scam': 'elderly_retiree',
        'phishing': 'elderly_retiree',
        'romance_scam': 'elderly_retiree',
        'tech_support_scam': 'elderly_retiree',
    }
    
    # Find matching persona
    for key, persona_name in persona_map.items():
        if key in scam_type_lower:
            return PERSONAS.get(persona_name, PERSONAS['elderly_retiree'])
    
    # Default to elderly retiree for unknown scam types
    return PERSONAS['elderly_retiree']


# ===================== LLM Integration =====================

def call_ollama(prompt: str, system_prompt: str = None, silent_errors: bool = False) -> str:
    """Call Ollama API for LLM response with retry logic."""
    import time
    
    for attempt in range(OLLAMA_MAX_RETRIES):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_predict": 80  # Reduced for faster responses
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                OLLAMA_URL, 
                json=payload, 
                timeout=OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                if not silent_errors:
                    print(f"Ollama error: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            if not silent_errors:
                print("⚠ Ollama not running. Start with: ollama serve")
            return None
        except requests.exceptions.Timeout:
            if attempt < OLLAMA_MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                if not silent_errors:
                    print(f"⏳ LLM timeout, retrying in {wait_time}s... (attempt {attempt + 2}/{OLLAMA_MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                if not silent_errors:
                    print("⚠ LLM timed out after all retries, using fallback.")
                return None
        except Exception as e:
            if not silent_errors:
                print(f"LLM error: {e}")
            return None
    
    return None


def call_ollama_streaming(prompt: str, system_prompt: str = None, print_output: bool = True) -> str:
    """Stream Ollama response for faster perceived performance.
    
    Prints tokens as they are generated, making responses feel instant.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 80
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            stream=True,
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code != 200:
            return None
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if print_output:
                        print(chunk, end="", flush=True)
                    full_response += chunk
                    
                    # Stop if done
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        if print_output:
            print()  # New line after streaming
        
        return full_response.strip()
        
    except requests.exceptions.ConnectionError:
        print("⚠ Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        print(f"Streaming error: {e}")
        return None

def analyze_message_context(message: str, metadata: dict = None, conversation_history: list = None) -> dict:
    """Use LLM to deeply understand the scam message context with full conversation history."""
    
    # Build conversation history context
    history_text = ""
    if conversation_history:
        for entry in conversation_history:
            history_text += f"SCAMMER: {entry['scammer']}\n"
            history_text += f"VICTIM: {entry['victim']}\n"
    
    system_prompt = """You are a scam analysis expert. Analyze the given conversation and latest message to extract:
1. What type of scam this is (may evolve as conversation progresses)
2. What the scammer currently wants (OTP, money, personal info, etc.)
3. The urgency tactics being used
4. Key claims being made
5. Recommended victim persona to engage them
6. Current conversation phase (initial contact, building trust, extraction attempt, etc.)

Consider the FULL conversation context to understand how the scam is progressing.
Respond in JSON format only."""

    prompt = f"""Analyze this scam conversation:

CONVERSATION HISTORY:
{history_text if history_text else 'This is the first message.'}

LATEST SCAMMER MESSAGE: {message}

METADATA: {json.dumps(metadata) if metadata else 'None'}

Respond with JSON only:
{{
    "scam_type": "type of scam",
    "scammer_goal": "what they currently want from victim",
    "urgency_tactics": ["list of tactics"],
    "false_claims": ["list of claims"],
    "recommended_persona": "elderly_retiree/college_student/busy_professional/small_business_owner",
    "key_topics": ["topics to engage on"],
    "conversation_phase": "current phase of scam",
    "scammer_frustration_level": "low/medium/high based on conversation"
}}"""

    response = call_ollama(prompt, system_prompt)
    
    if response:
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback to rule-based analysis
    return fallback_analyze(message)


def fallback_analyze(message: str) -> dict:
    """Rule-based fallback when LLM is unavailable."""
    message_lower = message.lower()
    
    scam_type = "unknown"
    scammer_goal = "unknown"
    persona = "elderly_retiree"
    
    if any(w in message_lower for w in ['otp', 'code', 'verify']):
        scam_type = "otp_scam"
        scammer_goal = "steal OTP code"
    elif any(w in message_lower for w in ['bank', 'account', 'blocked', 'suspend', 'kyc']):
        scam_type = "bank_phishing"
        scammer_goal = "steal banking credentials"
    elif any(w in message_lower for w in ['won', 'lottery', 'prize', 'winner', 'congratulation']):
        scam_type = "lottery_scam"
        scammer_goal = "collect processing fee"
        persona = "college_student"
    elif any(w in message_lower for w in ['job', 'earn', 'salary', 'work from home', 'hiring']):
        scam_type = "job_fraud"
        scammer_goal = "collect registration fee"
        persona = "college_student"
    elif any(w in message_lower for w in ['bitcoin', 'crypto', 'invest', 'trading', 'profit']):
        scam_type = "crypto_scam"
        scammer_goal = "steal investment"
        persona = "busy_professional"
    elif any(w in message_lower for w in ['loan', 'credit', 'emi', 'pre-approved']):
        scam_type = "loan_scam"
        scammer_goal = "collect processing fee"
        persona = "small_business_owner"
    
    return {
        "scam_type": scam_type,
        "scammer_goal": scammer_goal,
        "urgency_tactics": ["urgency detected"] if any(w in message_lower for w in ['urgent', 'immediate', 'now', 'quick']) else [],
        "false_claims": [],
        "recommended_persona": persona,
        "key_topics": []
    }


def generate_contextual_response(
    scammer_message: str,
    persona: Persona,
    context: dict,
    conversation_history: list,
    mode: BotMode
) -> str:
    """Generate a contextually appropriate victim response using LLM."""
    
    # Build conversation context
    history_text = ""
    for entry in conversation_history[-8:]:  # Last 8 turns for better context
        history_text += f"SCAMMER: {entry['scammer']}\n"
        history_text += f"VICTIM: {entry['victim']}\n"
    
    # Build the system prompt based on persona and mode
    system_prompt = f"""You are roleplaying as {persona.name}, a {persona.age}-year-old {persona.occupation}.

BACKGROUND: {persona.background}

SPEAKING STYLE: {persona.speaking_style}

VULNERABILITIES: {', '.join(persona.vulnerabilities)}

THINGS YOU DON'T UNDERSTAND: {', '.join(persona.knowledge_gaps)}

YOUR GOAL: Keep the scammer engaged and waste their time. You are NOT actually falling for the scam - you are pretending to be a naive victim. 

CURRENT MODE: {mode.value}
- If NAIVE_VICTIM: Act confused but willing, ask clarifying questions, seem to comply slowly
- If TIME_WASTING: Create delays - someone at door, phone dying, need to find glasses, etc.
- If INFO_EXTRACTION: Subtly ask for scammer's details - name, employee ID, callback number, bank name
- If CONFUSED: Misunderstand instructions, ask them to repeat, mix up details

IMPORTANT RULES:
1. Never actually give real OTPs or passwords
2. Give fake/wrong information when asked for sensitive data
3. Keep responses SHORT (1-2 sentences max)
4. Stay in character
5. Reference the specific context of what scammer is saying
6. After extracting some info, go back to time wasting"""

    prompt = f"""SCAM CONTEXT:
- Type: {context.get('scam_type', 'unknown')}
- Scammer wants: {context.get('scammer_goal', 'unknown')}
- Urgency tactics: {', '.join(context.get('urgency_tactics', []))}

CONVERSATION SO FAR:
{history_text}

LATEST SCAMMER MESSAGE: {scammer_message}

Generate {persona.name}'s response (1-2 sentences, stay in character):"""

    response = call_ollama(prompt, system_prompt)
    
    if response:
        # Clean up response
        response = response.strip()
        # Remove any meta-commentary
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        return response
    
    # Fallback responses if LLM fails - pass context for contextual responses
    return generate_fallback_response(mode, persona, scammer_message, context, conversation_history)


def generate_fallback_response(mode: BotMode, persona: Persona, message: str, context: dict = None, conversation_history: list = None) -> str:
    """Generate contextual fallback response without LLM.
    
    Uses scam type and conversation history to pick appropriate responses.
    """
    import random
    message_lower = message.lower()
    
    # Detect scam type from context or message
    scam_type = context.get('scam_type', 'unknown') if context else 'unknown'
    
    # If we have conversation history, detect context from it
    if scam_type == 'unknown' and conversation_history:
        full_convo = " ".join([h['scammer'] for h in conversation_history])
        full_convo_lower = full_convo.lower()
        if any(w in full_convo_lower for w in ['job', 'work from home', 'earn', 'salary', 'hiring']):
            scam_type = 'job_fraud'
        elif any(w in full_convo_lower for w in ['bank', 'account', 'blocked', 'kyc']):
            scam_type = 'bank_phishing'
        elif any(w in full_convo_lower for w in ['lottery', 'prize', 'won', 'winner']):
            scam_type = 'lottery_scam'
        elif any(w in full_convo_lower for w in ['otp', 'code', 'verify']):
            scam_type = 'otp_scam'
    
    # Context-aware responses based on scam type
    if scam_type == 'job_fraud':
        if mode == BotMode.TIME_WASTING:
            fallbacks = [
                "Wait, I need to ask my husband about this job opportunity...",
                "Hold on, can you tell me more about the company? I want to note it down...",
                "One second, I'm writing down the details... what was the salary again?",
                "Let me get a pen, I want to write down the website properly...",
            ]
        elif mode == BotMode.INFO_EXTRACTION:
            fallbacks = [
                "What is your company name? I want to research it first...",
                "Can you give me your contact number so I can call you back?",
                "What is the official company website? Not this link...",
                "Can you send me an email from the official company domain?",
            ]
        elif mode == BotMode.CONFUSED:
            fallbacks = [
                "₹25,000 per week? That seems very high... is this real?",
                "Work from home means what exactly? I don't understand...",
                "I need to do what on this link? Can you explain step by step?",
                "My nephew said don't click unknown links... but this is real job?",
            ]
        else:  # NAIVE_VICTIM
            fallbacks = [
                "Oh wow, ₹25,000 per week! I really need money for college fees...",
                "This job sounds perfect! I'm a student, I can work from home...",
                "No experience required? That's good because I just graduated...",
                "Limited slots? I should apply fast! What do I need to do?",
            ]
    
    elif scam_type == 'bank_phishing' or scam_type == 'otp_scam':
        if mode == BotMode.TIME_WASTING:
            fallbacks = [
                "Wait, someone is calling on my door... one minute...",
                "Hold on, let me find my reading glasses to see the message...",
                "One second, my phone screen is not showing properly...",
                "Let me sit down first, I'm feeling dizzy hearing this news...",
            ]
        elif mode == BotMode.INFO_EXTRACTION:
            fallbacks = [
                "Before anything, which branch are you calling from exactly?",
                "What is your employee ID? I need to verify you are from bank...",
                "Can you give me a number to call you back on?",
                "What is your full name? I want to complaint to bank manager...",
            ]
        elif mode == BotMode.CONFUSED:
            fallbacks = [
                "Which account? I have SBI, HDFC... which one is blocked?",
                "OTP? What is OTP? You mean the SMS code?",
                "I don't understand... my account is blocked or suspended?",
                "Should I call my son? He handles all banking for me...",
            ]
        else:  # NAIVE_VICTIM
            fallbacks = [
                "Oh no! Please help me, all my savings are in that account!",
                "My account is blocked? What did I do wrong? Please fix it!",
                "I got some code on my phone... is this what you need?",
                "I don't want my money to go! Tell me what to do!",
            ]
    
    elif scam_type == 'lottery_scam':
        if mode == BotMode.TIME_WASTING:
            fallbacks = [
                "Wait, let me tell my wife about this... she won't believe it!",
                "Hold on, I need to find a paper to write down the prize amount...",
                "One second, my grandchildren are here, I want to share good news...",
            ]
        elif mode == BotMode.INFO_EXTRACTION:
            fallbacks = [
                "Which lottery is this? I don't remember entering any...",
                "What is your office address? I want to come collect prize...",
                "Can you send me official documents by post?",
            ]
        else:
            fallbacks = [
                "I won a lottery? This is the best day of my life!",
                "How much did I win? I can't believe this is happening!",
                "My prayers have been answered! Tell me how to claim!",
            ]
    
    else:  # Generic fallbacks for unknown scam types
        if mode == BotMode.TIME_WASTING:
            fallbacks = [
                "Wait one minute, someone is at the door...",
                "Hold on, let me find my reading glasses...",
                "One second, I'm getting another call...",
            ]
        elif mode == BotMode.INFO_EXTRACTION:
            fallbacks = [
                "Before that, can you tell me your name and ID?",
                "Which company are you from exactly?",
                "Can you give me a number to call you back?",
            ]
        elif mode == BotMode.CONFUSED:
            fallbacks = [
                "What? I didn't understand. Can you say again slowly?",
                "I'm confused... can you explain step by step?",
                "My son usually helps me with this. Should I call him?",
            ]
        else:
            fallbacks = [
                "Yes, yes, I'm listening. What do I need to do?",
                "Okay, please help me. I don't know what's happening...",
                "Haan ji, please guide me... I'm worried now...",
            ]
    
    return random.choice(fallbacks)


# ===================== Main Bot Class =====================

class ScamBaitingBot:
    """LLM-powered scam baiting chatbot."""
    
    def __init__(self):
        self.persona: Optional[Persona] = None
        self.context: dict = {}
        self.mode = BotMode.NAIVE_VICTIM
        self.conversation_history: list = []
        self.conversation_phase = "initial_contact"  # Track conversation phase
        self.extracted_intel = {
            'urls': [],
            'phone_numbers': [],
            'names': [],
            'organizations': [],
            'email_addresses': [],
            'account_numbers': [],
        }
        self.turn_count = 0
        self.llm_available = True
        
    def update_context(self, message: str, metadata: dict = None) -> None:
        """Update context analysis based on full conversation history.
        
        This is called on EVERY turn to iteratively understand the conversation.
        """
        # Get LLM analysis with full conversation history
        self.context = analyze_message_context(
            message=message, 
            metadata=metadata, 
            conversation_history=self.conversation_history
        )
        
        # Update conversation phase based on turn count
        if self.turn_count <= 2:
            self.conversation_phase = "initial_contact"
        elif self.turn_count <= 5:
            self.conversation_phase = "building_trust"
        elif self.turn_count <= 10:
            self.conversation_phase = "extraction_attempt"
        else:
            self.conversation_phase = "extended_engagement"
        
        # Set persona on first message using smart selection based on scam type
        if self.turn_count == 1:
            scam_type = self.context.get('scam_type', 'unknown')
            self.persona = select_persona_for_scam(scam_type)
            
            # Check if LLM is available on first message
            test_response = call_ollama("test", "respond with 'ok'", silent_errors=True)
            self.llm_available = test_response is not None
            
            if not self.llm_available:
                print("⚠ LLM not available. Using rule-based responses.")
    
    def extract_intel_from_message(self, message: str) -> None:
        """Extract any useful intelligence from scammer's message."""
        intel = extract_threat_intelligence(message)
        
        # Add unique items
        for url in intel.get('urls', []):
            if url not in self.extracted_intel['urls']:
                self.extracted_intel['urls'].append(url)
        
        for phone in intel.get('phones', []):
            if phone not in self.extracted_intel['phone_numbers']:
                self.extracted_intel['phone_numbers'].append(phone)
        
        for org in intel.get('organizations', []):
            if org not in self.extracted_intel['organizations']:
                self.extracted_intel['organizations'].append(org)
        
        for email in intel.get('emails', []):
            if email not in self.extracted_intel['email_addresses']:
                self.extracted_intel['email_addresses'].append(email)
    
    def decide_mode(self) -> BotMode:
        """Decide which mode to operate in for this turn."""
        import random
        
        # First few turns - be naive
        if self.turn_count <= 2:
            return BotMode.NAIVE_VICTIM
        
        # Every 4th turn, try to extract info
        if self.turn_count % 4 == 0:
            return BotMode.INFO_EXTRACTION
        
        # Every 3rd turn, waste time
        if self.turn_count % 3 == 0:
            return BotMode.TIME_WASTING
        
        # Occasionally be confused
        if random.random() < 0.2:
            return BotMode.CONFUSED
        
        return BotMode.NAIVE_VICTIM
    
    def generate_response(self, scammer_message: str, metadata: dict = None) -> dict:
        """Generate a response to the scammer's message.
        
        Returns None if message is detected as genuine (not spam).
        """
        # First, check if this is actually a spam/scam message
        is_spam, spam_confidence = is_spam_message(scammer_message, SPAM_THRESHOLD, metadata)
        
        if not is_spam:
            # This is a genuine message - don't respond
            return {
                'response': None,
                'is_spam': False,
                'spam_confidence': spam_confidence,
                'message': 'Message detected as genuine - no response needed',
                'turn': self.turn_count
            }
        
        # It's spam/scam - engage!
        self.turn_count += 1
        
        # Update context on EVERY turn to understand conversation evolution
        self.update_context(scammer_message, metadata)
        
        # Extract intel from message
        self.extract_intel_from_message(scammer_message)
        
        # Decide mode
        self.mode = self.decide_mode()
        
        # Generate response - use fast fallback for first 2 turns
        if USE_FAST_MODE and self.turn_count <= 2:
            # Instant response using fallback - no LLM call needed
            response = generate_fallback_response(
                self.mode, self.persona, scammer_message, 
                self.context, self.conversation_history
            )
        elif self.llm_available:
            response = generate_contextual_response(
                scammer_message=scammer_message,
                persona=self.persona,
                context=self.context,
                conversation_history=self.conversation_history,
                mode=self.mode
            )
        else:
            response = generate_fallback_response(
                self.mode, self.persona, scammer_message, 
                self.context, self.conversation_history
            )
        
        # Store in history
        self.conversation_history.append({
            'turn': self.turn_count,
            'scammer': scammer_message,
            'victim': response,
            'mode': self.mode.value
        })
        
        return {
            'response': response,
            'persona': self.persona.name,
            'persona_details': {
                'age': self.persona.age,
                'occupation': self.persona.occupation,
                'background': self.persona.background[:100] + '...'
            },
            'mode': self.mode.value,
            'conversation_phase': self.conversation_phase,
            'scam_analysis': self.context,
            'turn': self.turn_count,
            'extracted_intel': self.extracted_intel,
            'llm_used': self.llm_available
        }
    
    def get_conversation_summary(self) -> dict:
        """Get conversation summary with extracted intel."""
        return {
            'total_turns': self.turn_count,
            'persona_used': self.persona.name if self.persona else "Unknown",
            'persona_occupation': self.persona.occupation if self.persona else "Unknown",
            'scam_type': self.context.get('scam_type', 'unknown'),
            'scammer_goal': self.context.get('scammer_goal', 'unknown'),
            'final_phase': self.conversation_phase,
            'time_wasted_estimate': f"{self.turn_count * 2} minutes",
            'extracted_intelligence': self.extracted_intel,
            'conversation_history': self.conversation_history,
            'llm_used': self.llm_available
        }


# ===================== CLI Interface =====================

def run_cli():
    """Interactive CLI for testing the bot."""
    print("\n" + "=" * 70)
    print("🤖 SCAM BAITING BOT - LLM-Powered Anti-Scam AI")
    print("=" * 70)
    print("\nThis bot PRETENDS to be a victim to waste scammer time.")
    print("You play the SCAMMER, the bot plays a naive VICTIM.")
    print("\nFeatures:")
    print("  • Analyzes your message to understand scam context")
    print("  • Selects appropriate victim persona automatically")
    print("  • Uses LLM to generate contextual responses")
    print("  • Extracts intel (URLs, phones, etc.) from your messages")
    print("\nType 'quit' to end. Type 'info' to see extracted intel.")
    print("-" * 70)
    
    bot = ScamBaitingBot()
    
    while True:
        try:
            user_input = input("\n😈 SCAMMER: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = 'quit'
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'info':
            print(f"\n📊 Extracted Intel: {json.dumps(bot.extracted_intel, indent=2)}")
            continue
        
        # Generate response
        result = bot.generate_response(user_input)
        
        if result['response'] is None:
            # Genuine message
            print(f"\n✅ SYSTEM: {result['message']}")
            print(f"   Spam Confidence: {result['spam_confidence']:.2f}")
            continue
            
        print(f"\n🎭 VICTIM [{result['persona']}]: {result['response']}")
        print(f"   Mode: {result['mode']} | Phase: {result['conversation_phase']} | LLM: {'✓' if result['llm_used'] else '✗'} | Turn: {result['turn']}")
        
        if result['turn'] == 1:
            print(f"   Persona: {result['persona_details']['occupation']} ({result['persona_details']['age']} years old)")
            print(f"   Scam Type: {result['scam_analysis'].get('scam_type', 'unknown')}")
            print(f"   Scammer Goal: {result['scam_analysis'].get('scammer_goal', 'unknown')}")
    
    # Show summary
    summary = bot.get_conversation_summary()
    print("\n" + "=" * 70)
    print("📊 CONVERSATION SUMMARY")
    print("=" * 70)
    print(f"  Persona: {summary['persona_used']} ({summary['persona_occupation']})")
    print(f"  Scam Type: {summary['scam_type']}")
    print(f"  Scammer Goal: {summary['scammer_goal']}")
    print(f"  Final Phase: {summary['final_phase']}")
    print(f"  Turns: {summary['total_turns']}")
    print(f"  Time Wasted: {summary['time_wasted_estimate']}")
    print(f"  LLM Used: {'Yes' if summary['llm_used'] else 'No (fallback)'}")
    
    if summary['extracted_intelligence']['urls']:
        print(f"\n  🔗 URLs: {summary['extracted_intelligence']['urls']}")
    if summary['extracted_intelligence']['phone_numbers']:
        print(f"  📱 Phones: {summary['extracted_intelligence']['phone_numbers']}")
    if summary['extracted_intelligence']['organizations']:
        print(f"  🏢 Orgs: {summary['extracted_intelligence']['organizations']}")
    
    print("=" * 70 + "\n")


# ===================== Exports =====================

def create_bot() -> ScamBaitingBot:
    """Create a new bot instance."""
    return ScamBaitingBot()

if __name__ == "__main__":
    run_cli()
