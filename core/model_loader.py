"""Load and cache ML models for spam and scam classification."""

import os
import joblib

from config import MODEL_DIR


class ModelCache:
    """Global cache for loaded models."""
    spam_vectorizer = None
    spam_model = None
    scam_vectorizer = None
    scam_model = None
    last_analysis = None


def load_models() -> None:
    """Load ML models into memory."""
    try:
        spam_vec_path = os.path.join(MODEL_DIR, "spam_vectorizer.joblib")
        spam_model_path = os.path.join(MODEL_DIR, "spam_detector.joblib")
        scam_vec_path = os.path.join(MODEL_DIR, "scam_vectorizer.joblib")
        scam_model_path = os.path.join(MODEL_DIR, "scam_classifier.joblib")

        if os.path.exists(spam_vec_path) and os.path.exists(spam_model_path):
            ModelCache.spam_vectorizer = joblib.load(spam_vec_path)
            ModelCache.spam_model = joblib.load(spam_model_path)
            print("✓ Spam detection models loaded")
        else:
            print("⚠ Spam models not found. Run training first.")

        if os.path.exists(scam_vec_path) and os.path.exists(scam_model_path):
            ModelCache.scam_vectorizer = joblib.load(scam_vec_path)
            ModelCache.scam_model = joblib.load(scam_model_path)
            print("✓ Scam classification models loaded")
        else:
            print("⚠ Scam models not found. Run training first.")

    except Exception as e:
        print(f"⚠ Error loading models: {e}")
