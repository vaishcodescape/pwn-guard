"""
Scam Type Classification Training Module

Trains a TF-IDF + Linear SVM classifier for multi-class scam type classification.
Only runs when spam is detected.

Scam Types:
- bank_phishing
- otp_scam
- job_fraud
- crypto_scam
- lottery_scam
- loan_scam
- other
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from config import DATA_DIR, MODEL_DIR
from extraction.preprocess import preprocess_message


# Scam type definitions
SCAM_TYPES = [
    'bank_phishing',
    'otp_scam',
    'job_fraud',
    'crypto_scam',
    'lottery_scam',
    'loan_scam',
    'other'
]


def create_scam_type_dataset() -> pd.DataFrame:
    """
    Create a labeled dataset for scam type classification.
    In production, this would be replaced with a manually curated dataset.
    """
    data = {
        'text': [],
        'scam_type': []
    }

    # Bank phishing examples
    bank_phishing = [
        "Your SBI account has been compromised. Verify your details at sbi-verify.fake.com",
        "ALERT: Unauthorized access to your HDFC account. Update now: hdfc-update.com",
        "Dear customer, your bank account will be blocked. Click to verify: bank-verify.com",
        "Urgent: Your ICICI credentials need verification. Login now: icici-secure.fake.com",
        "Your bank card has been suspended. Reactivate: card-reactivate.com",
        "Security alert: Unusual activity detected on your account. Verify identity now",
        "Your net banking password expires today. Update immediately to continue services",
        "Final warning: Your account will be closed unless you verify at bank-final.com",
        "ATM card blocked due to suspicious activity. Unblock now: atm-unblock.com",
        "Bank of India: Your account is under review. Submit documents: boi-docs.fake.com",
    ]

    # OTP scam examples
    otp_scam = [
        "Your OTP is 654321. Share this to verify your payment of Rs.50,000",
        "Transaction of Rs.25,000 initiated. Share OTP 123456 to cancel",
        "Paytm: Enter OTP 789012 to complete verification. Do not share with anyone",
        "Your WhatsApp OTP is 456789. Give this to our agent to restore your account",
        "To block your stolen card, provide the OTP you just received",
        "Google Pay verification: Share the code we sent to confirm your identity",
        "Amazon order requires OTP verification. Tell us the code to process delivery",
        "PhonePe: Your OTP for Rs.10,000 transfer is pending. Confirm with code",
        "URGENT: Share OTP immediately to stop unauthorized transaction",
        "Your Netflix OTP: 234567. Provide to customer care to fix billing issue",
    ]

    # Job fraud examples
    job_fraud = [
        "Earn $5000/week working from home! No experience needed. Apply now",
        "Part-time job offer: Make Rs.50,000 monthly. Just 2 hours work daily",
        "Data entry job: Earn $100/hour from home. No interview required",
        "Hiring! Work from home and earn $3000 weekly. Simple typing tasks",
        "Exclusive job offer from Google. Salary $200,000. Reply with resume",
        "Amazon hiring for remote positions. $50/hour guaranteed. Register now",
        "Work from anywhere! Earn Rs.1,00,000 monthly. Registration fee: Rs.500",
        "Online job opportunity: $500/day for simple surveys. No experience needed",
        "Attention job seekers! Immediate hiring. High salary. No qualification required",
        "MNC company hiring freshers. Package: Rs.50 LPA. Apply with fee of Rs.1000",
    ]

    # Crypto scam examples
    crypto_scam = [
        "Double your Bitcoin in 24 hours! Guaranteed returns. Invest now",
        "You've received 0.5 BTC! Claim at crypto-claim.com before it expires",
        "Elon Musk is giving away Ethereum! Send 0.1 ETH, get 1 ETH back",
        "New cryptocurrency launching. Invest $100, get $10,000 in 30 days",
        "Bitcoin trading bot: 500% returns daily. No risk, guaranteed profit",
        "Crypto airdrop: Free 1 BTC for first 1000 users. Register now",
        "NFT giveaway! Free valuable NFT. Just connect your wallet",
        "Blockchain investment opportunity. 10x returns in 1 week. Act now!",
        "Your Coinbase account suspended. Verify at coinbase-verify.fake.com",
        "Binance security alert: Claim your free crypto before account closure",
    ]

    # Lottery scam examples
    lottery_scam = [
        "CONGRATULATIONS! You've won $1,000,000 in the international lottery!",
        "You are the lucky winner of Rs.50,00,000 in KBC Kaun Banega Crorepati",
        "Your number selected for prize of Rs.25 lakhs. Claim now!",
        "WINNER! iPhone 15 Pro Max is yours. Just pay shipping fee",
        "Lucky draw winner! You've won a Mercedes Benz. Contact to claim",
        "Congratulations! Selected for Rs.1 crore prize. Send Aadhar to verify",
        "MEGA JACKPOT! You won $500,000. Pay processing fee to receive",
        "Diwali bumper: Your ticket won Rs.10 lakhs. Claim immediately",
        "BBC international lottery: You won £850,000. Reply with bank details",
        "Toyota car lottery winner! You won a new Fortuner. Claim at lottery-claim.com",
    ]

    # Loan scam examples
    loan_scam = [
        "Pre-approved loan of Rs.10,00,000 at 0% interest. Apply now!",
        "Instant loan approved! Get Rs.5,00,000 in 5 minutes. No documents needed",
        "Low credit score? Get loan anyway! Guaranteed approval. Apply today",
        "Personal loan at just 1% interest. No security required. Fast approval",
        "URGENT: Your loan application approved. Pay processing fee to receive",
        "Government scheme: Interest-free loan for all. Limited time offer",
        "Loan settlement offer: Pay 50% and clear your entire loan. Act now",
        "Bad credit? No problem! Get $50,000 loan instantly. Apply here",
        "Home loan at 0% down payment. No income proof needed. Apply now",
        "Student loan forgiveness: Clear your debt for just $500. Limited offer",
    ]

    # Other scams (general/misc)
    other_scams = [
        "Your computer has a virus! Call Microsoft support immediately",
        "IRS notice: You owe taxes. Pay now or face arrest",
        "Your package is stuck in customs. Pay $50 to release",
        "Install this app to get free mobile recharge forever",
        "Nigerian prince needs help transferring $10 million",
        "Your social security number has been compromised. Call now",
        "Hot singles in your area want to meet you! Click here",
        "Free vacation to Maldives! Just complete this survey",
        "Weight loss miracle: Lose 30 pounds in 1 week. Order now",
        "Psychic reading: I see great fortune in your future. Call now",
    ]

    # Add all to dataset
    for msg in bank_phishing:
        data['text'].append(msg)
        data['scam_type'].append('bank_phishing')

    for msg in otp_scam:
        data['text'].append(msg)
        data['scam_type'].append('otp_scam')

    for msg in job_fraud:
        data['text'].append(msg)
        data['scam_type'].append('job_fraud')

    for msg in crypto_scam:
        data['text'].append(msg)
        data['scam_type'].append('crypto_scam')

    for msg in lottery_scam:
        data['text'].append(msg)
        data['scam_type'].append('lottery_scam')

    for msg in loan_scam:
        data['text'].append(msg)
        data['scam_type'].append('loan_scam')

    for msg in other_scams:
        data['text'].append(msg)
        data['scam_type'].append('other')

    return pd.DataFrame(data)


def train_scam_classifier():
    """Train the scam type classification model."""
    print("=" * 60)
    print("SCAM TYPE CLASSIFICATION MODEL TRAINING")
    print("=" * 60)

    # Load or create dataset
    csv_path = os.path.join(DATA_DIR, 'scam_types.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} scam examples from {csv_path}")
    else:
        print("Creating scam type training dataset...")
        df = create_scam_type_dataset()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")

    # Preprocess messages
    print("\nPreprocessing messages...")
    processed_texts = []
    for text in df['text']:
        result = preprocess_message(text)
        processed_texts.append(result['processed_text'])

    df['processed_text'] = processed_texts

    print(f"\nScam type distribution:")
    print(df['scam_type'].value_counts())

    # Split data
    X = df['processed_text']
    y = df['scam_type']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Create TF-IDF vectorizer
    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words='english'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Train Linear SVM with probability calibration
    print("\nTraining Linear SVM classifier...")
    base_svm = LinearSVC(
        random_state=42,
        class_weight='balanced',
        max_iter=2000,
        C=1.0
    )

    # Wrap with calibration to get probability estimates
    model = CalibratedClassifierCV(base_svm, cv=3)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test_tfidf)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    vectorizer_path = os.path.join(MODEL_DIR, 'scam_vectorizer.joblib')
    model_path = os.path.join(MODEL_DIR, 'scam_classifier.joblib')

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Model saved to: {model_path}")

    print("\n✓ Training complete!")

    return vectorizer, model


def predict_scam_type(text: str, vectorizer=None, model=None) -> dict:
    """
    Predict the scam type of a spam message.

    Args:
        text: Message text
        vectorizer: TF-IDF vectorizer (loads from disk if None)
        model: Trained model (loads from disk if None)

    Returns:
        Dictionary with prediction results
    """
    if vectorizer is None or model is None:
        vectorizer_path = os.path.join(MODEL_DIR, 'scam_vectorizer.joblib')
        model_path = os.path.join(MODEL_DIR, 'scam_classifier.joblib')

        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Models not found. Run training first.")

        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)

    # Preprocess
    processed = preprocess_message(text)

    # Vectorize
    X = vectorizer.transform([processed['processed_text']])

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Get class probabilities
    classes = model.classes_
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    return {
        'scam_type': prediction,
        'confidence': float(max(probabilities)),
        'probabilities': prob_dict,
    }


if __name__ == "__main__":
    train_scam_classifier()

    # Test predictions
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    test_messages = [
        "Your SBI account is blocked. Verify at fake-sbi.com",
        "Share your OTP 123456 to cancel the transaction",
        "Work from home and earn $5000 daily!",
        "You've won 1 Bitcoin! Claim now!",
        "CONGRATULATIONS! You won Rs.50 lakhs in KBC!",
        "Pre-approved loan of Rs.10 lakhs at 0% interest",
    ]

    for msg in test_messages:
        result = predict_scam_type(msg)
        print(f"\nMessage: {msg[:50]}...")
        print(f"  Type: {result['scam_type']} (confidence: {result['confidence']:.2%})")
