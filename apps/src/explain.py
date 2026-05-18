import numpy as np


RISK_WHEN_HIGH = {
    "HighBP",
    "GenHlth",
    "HighChol",
    "Age_BMI_Risk",
    "BMI",
    "Smoker",
    "HvyAlcoholConsump",
}

PROTECTIVE_WHEN_HIGH = {"PhysActivity", "CholCheck"}


def get_top_risk_factors(shap_vals_single, feature_names, n=3):
    """
    Ambil N fitur dengan SHAP value absolut terbesar.
    """
    idx_sorted = np.argsort(np.abs(shap_vals_single))[::-1]
    hasil = []
    for i in idx_sorted[:n]:
        hasil.append(
            {
                "feature": feature_names[i],
                "shap_value": float(shap_vals_single[i]),
                "direction": "risk" if shap_vals_single[i] > 0 else "protective",
            }
        )
    return hasil


def get_top_fallback_factors(scaled_vals_single, feature_names, n=3):
    """
    Fallback saat SHAP explainer belum tersedia.

    Nilainya bukan SHAP asli; ini memakai besarnya nilai input yang sudah
    distandardisasi agar endpoint tetap bisa memberi faktor dominan.
    """
    idx_sorted = np.argsort(np.abs(scaled_vals_single))[::-1]
    hasil = []
    for i in idx_sorted[:n]:
        feature = feature_names[i]
        value = float(scaled_vals_single[i])

        if feature in PROTECTIVE_WHEN_HIGH:
            direction = "protective" if value > 0 else "risk"
        elif feature in RISK_WHEN_HIGH:
            direction = "risk" if value > 0 else "protective"
        else:
            direction = "risk" if value > 0 else "protective"

        hasil.append(
            {
                "feature": feature,
                "shap_value": value,
                "direction": direction,
            }
        )
    return hasil
