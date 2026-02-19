"""
Spam Detection Training Module

Trains a TF-IDF + Logistic Regression classifier for binary spam detection.
Run from project root: python -m training.train_spam
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import DATA_DIR, MODEL_DIR
from extraction.preprocess import preprocess_message


def load_or_create_dataset() -> pd.DataFrame:
    """
    Load SMS Spam Collection dataset or create sample data.
    Dataset: data/spam.csv with columns label ('spam'/'ham') and text.
    """
    csv_path = os.path.join(DATA_DIR, "spam.csv")

    if os.path.exists(csv_path):
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                if "v1" in df.columns and "v2" in df.columns:
                    df = df.rename(columns={"v1": "label", "v2": "text"})
                    df = df[["label", "text"]]
                print(f"Loaded {len(df)} messages from {csv_path}")
                return df
            except Exception:
                continue
        print(f"Warning: Could not read {csv_path}. Using sample data.")

    # Create sample dataset
    print("Creating sample training dataset...")
    sample_data = {"label": [], "text": []}

    spam_messages = [
        "URGENT! You have won $1,000,000! Click http://fakeprize.com to claim NOW!",
        "Congratulations! You've been selected for a free iPhone. Visit bit.ly/freeiphone",
        "Your bank account has been compromised. Verify now: http://fake-bank.com",
        "WINNER! You've won a $500 gift card. Claim here: http://giftcard-scam.com",
        "Make $5000 daily from home! No experience needed. Reply YES",
        "Dear customer, your SBI account will be blocked. Update KYC: sbi-update.fake.com",
        "You've been pre-approved for a loan of $50,000! Act now before offer expires",
        "CONGRATS! Your number selected for Rs.25 lakhs prize. Send Aadhar to claim",
    ]

    ham_messages = [
        "Hey, are we still meeting for coffee tomorrow at 3pm?",
        "Can you pick up some groceries on your way home?",
        "Happy birthday! Hope you have a great day!",
        "The meeting has been rescheduled to 4pm today",
        "Your verification code is 1234. Do not share it.",
        "Your package from Amazon has been delivered.",
        "A/c *1464 Debited for Rs:1500 on 05-02-2026. Avl Bal Rs:210.",
    ]

    for msg in spam_messages:
        sample_data["label"].append("spam")
        sample_data["text"].append(msg)
    for msg in ham_messages:
        sample_data["label"].append("ham")
        sample_data["text"].append(msg)

    df = pd.DataFrame(sample_data)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Sample dataset saved to {csv_path}")
    return df


def train_spam_detector():
    """Train the spam detection model."""
    print("=" * 60)
    print("SPAM DETECTION MODEL TRAINING")
    print("=" * 60)

    df = load_or_create_dataset()

    print("\nPreprocessing messages...")
    processed_texts = [preprocess_message(text)["processed_text"] for text in df["text"]]
    df["processed_text"] = processed_texts
    df["spam"] = (df["label"] == "spam").astype(int)

    X = df["processed_text"]
    y = df["spam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english"
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        C=1.0
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "spam_vectorizer.joblib"))
    joblib.dump(model, os.path.join(MODEL_DIR, "spam_detector.joblib"))
    print("\n✓ Models saved to", MODEL_DIR)
    return vectorizer, model


if __name__ == "__main__":
    train_spam_detector()
