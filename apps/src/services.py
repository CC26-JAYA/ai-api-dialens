import json
import os

import joblib
import tensorflow as tf

from src.model import HealthClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP \
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

FEATURES = [
    "HighBP", "GenHlth", "HighChol", "Age_BMI_Risk",
    "CholCheck", "HvyAlcoholConsump", "BMI", "PhysActivity", "Smoker",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
model = HealthClassifier(input_dim=9)
_ = model(tf.zeros((1, 9)), training=False)

try:
    model.load_weights(os.path.join(ARTIFACTS_DIR, "best_weights.weights.h5"))
except Exception as e:
    print(f"Warning: Could not load weights: {e}")

try:
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_inference.pkl"))
except Exception as e:
    print(f"Warning: Could not load scaler: {e}")
    scaler = None

try:
    explainer = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"))
except Exception as e:
    print(f"Warning: Could not load explainer: {e}")
    explainer = None

with open(os.path.join(ARTIFACTS_DIR, "history.json")) as f:
    _history = json.load(f)
THRESHOLD = _history.get("best_threshold", 0.5)

print(f"model loaded, threshold={THRESHOLD:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def label_risk(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Moderate"
    return "High"
