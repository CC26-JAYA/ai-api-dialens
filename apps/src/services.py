import json
import importlib.util
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

from .explain import get_top_fallback_factors, get_top_risk_factors
from .model import HealthClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP \
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

FEATURES = [
    "HighBP",
    "GenHlth",
    "HighChol",
    "Age_BMI_Risk",
    "CholCheck",
    "HvyAlcoholConsump",
    "BMI",
    "PhysActivity",
    "Smoker",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
class ArtifactNotReadyError(RuntimeError):
    pass


_artifact_errors = {
    "model": None,
    "scaler": None,
    "explainer": None,
    "threshold": None,
}
_artifact_warnings = {
    "model": [],
    "scaler": [],
    "explainer": [],
    "threshold": [],
}
_runtime_kernel_explainer = None
_runtime_explainer_error = None


def _load_joblib_artifact(path: Path, artifact_name: str):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            artifact = joblib.load(path)
        _artifact_warnings[artifact_name].extend(str(item.message) for item in caught)
        return artifact
    except Exception as e:
        _artifact_errors[artifact_name] = str(e)
        return None


model = HealthClassifier(input_dim=len(FEATURES))
_ = model(tf.zeros((1, len(FEATURES))), training=False)

try:
    model.load_weights(ARTIFACTS_DIR / "best_weights.weights.h5")
except Exception as e:
    _artifact_errors["model"] = str(e)

scaler = _load_joblib_artifact(ARTIFACTS_DIR / "scaler_inference.pkl", "scaler")
explainer = _load_joblib_artifact(ARTIFACTS_DIR / "shap_explainer.pkl", "explainer")
if explainer is not None and not hasattr(explainer, "shap_values"):
    _artifact_warnings["explainer"].append(
        f"Loaded {type(explainer).__name__} has no shap_values API; "
        "using runtime KernelExplainer instead."
    )

try:
    with open(ARTIFACTS_DIR / "history.json") as f:
        _history = json.load(f)
    THRESHOLD = float(_history.get("best_threshold", 0.5))
except Exception as e:
    _artifact_errors["threshold"] = str(e)
    THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _can_use_runtime_kernel_explainer():
    return (
        importlib.util.find_spec("shap") is not None
        and scaler is not None
        and _artifact_errors["model"] is None
        and (DATA_DIR / "data_test_unscaled.csv").exists()
    )


def _explainability_ready():
    return (explainer is not None and hasattr(explainer, "shap_values")) or (
        _runtime_explainer_error is None and _can_use_runtime_kernel_explainer()
    )


def _explainability_mode():
    if explainer is not None and hasattr(explainer, "shap_values"):
        return "artifact_shap_values"
    if _runtime_explainer_error is None and _can_use_runtime_kernel_explainer():
        return "runtime_kernel_shap"
    return "standardized_input_fallback"


def get_artifact_status():
    return {
        "model": {
            "ready": _artifact_errors["model"] is None,
            "error": _artifact_errors["model"],
            "warnings": _artifact_warnings["model"],
        },
        "scaler": {
            "ready": scaler is not None,
            "error": _artifact_errors["scaler"],
            "warnings": _artifact_warnings["scaler"],
        },
        "explainer": {
            "ready": _explainability_ready(),
            "mode": _explainability_mode(),
            "error": _runtime_explainer_error or _artifact_errors["explainer"],
            "warnings": _artifact_warnings["explainer"],
        },
        "threshold": {
            "ready": _artifact_errors["threshold"] is None,
            "error": _artifact_errors["threshold"],
            "warnings": _artifact_warnings["threshold"],
        },
    }


def ensure_inference_ready():
    status = get_artifact_status()
    missing = [
        name for name in ("model", "scaler", "threshold") if not status[name]["ready"]
    ]
    if missing:
        details = ", ".join(f"{name}: {status[name]['error']}" for name in missing)
        raise ArtifactNotReadyError(f"Inference artifacts are not ready ({details})")


def _predict_for_shap(values):
    values = np.asarray(values, dtype=np.float32)
    return model(values, training=False).numpy().reshape(-1)


def _get_runtime_kernel_explainer():
    global _runtime_kernel_explainer, _runtime_explainer_error
    if _runtime_kernel_explainer is not None:
        return _runtime_kernel_explainer

    try:
        import shap

        background = pd.read_csv(
            DATA_DIR / "data_test_unscaled.csv",
            usecols=FEATURES,
        ).head(20)
        background_scaled = scaler.transform(background).astype("float32")
        _runtime_kernel_explainer = shap.KernelExplainer(
            _predict_for_shap,
            background_scaled,
        )
        _runtime_explainer_error = None
        return _runtime_kernel_explainer
    except Exception as e:
        _runtime_explainer_error = str(e)
        return None


def explain_scaled_input(scaled_input):
    global _runtime_explainer_error

    if explainer is not None and hasattr(explainer, "shap_values"):
        sv = explainer.shap_values(scaled_input)
        if isinstance(sv, list):
            sv = sv[0]
        return get_top_risk_factors(sv[0], FEATURES), "shap"

    runtime_explainer = _get_runtime_kernel_explainer()
    if runtime_explainer is not None:
        try:
            sv = runtime_explainer.shap_values(
                scaled_input,
                nsamples=50,
                silent=True,
            )
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 0]
            return get_top_risk_factors(sv[0], FEATURES), "shap_kernel_runtime"
        except Exception as e:
            _runtime_explainer_error = str(e)

    return (
        get_top_fallback_factors(scaled_input[0], FEATURES),
        "standardized_input_fallback",
    )


def label_risk(prob: float) -> str:
    if prob < 0.45:
        return "Low"
    elif prob < 0.60:
        return "Moderate"
    return "High"
