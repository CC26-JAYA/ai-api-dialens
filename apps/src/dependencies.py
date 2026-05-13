import joblib
import tensorflow as tf
from src.model import HealthClassifier
import os

# Definisikan path ke file artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model = HealthClassifier(input_dim=9)
# Jalanin forward pass sekali dengan dummy data agar layer ter-build sebelum load weights
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
