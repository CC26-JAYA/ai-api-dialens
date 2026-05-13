import numpy as np


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
