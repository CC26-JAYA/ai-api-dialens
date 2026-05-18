import pandas as pd
from fastapi import APIRouter, HTTPException, Response, status

from .schemas import PatientInput, PredictResponse, FactorItem
from .services import (
    FEATURES,
    THRESHOLD,
    ArtifactNotReadyError,
    ensure_inference_ready,
    explain_scaled_input,
    get_artifact_status,
    label_risk,
    model,
    scaler,
)
from .llm import generate_recommendation

router = APIRouter()


@router.get("/health")
def health(response: Response):
    artifact_status = get_artifact_status()
    ready = (
        artifact_status["model"]["ready"]
        and artifact_status["scaler"]["ready"]
        and artifact_status["threshold"]["ready"]
    )
    status_text = (
        "ok" if ready and artifact_status["explainer"]["ready"] else "degraded"
    )
    if not ready:
        status_text = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": status_text,
        "ready": ready,
        "explainability_ready": artifact_status["explainer"]["ready"],
        "threshold": THRESHOLD,
        "params": int(model.count_params()) if model is not None else 0,
        "artifacts": artifact_status,
    }


@router.get("/features")
def features():
    return {
        "features": FEATURES,
        "note": "Age_BMI_Risk dihitung otomatis dari Age x BMI",
        "fields": {
            "HighBP": "0/1",
            "GenHlth": "1-5",
            "HighChol": "0/1",
            "Age": "1-13 (grup usia CDC)",
            "CholCheck": "0/1",
            "HvyAlcoholConsump": "0/1",
            "BMI": "float",
            "PhysActivity": "0/1",
            "Smoker": "0/1",
        },
    }


@router.post("/predict", response_model=PredictResponse)
def predict(p: PatientInput):
    try:
        ensure_inference_ready()

        # 1. feature engineering
        age_bmi = p.Age * p.BMI

        # 2. scaling
        raw = pd.DataFrame(
            [
                [
                    p.HighBP,
                    p.GenHlth,
                    p.HighChol,
                    age_bmi,
                    p.CholCheck,
                    p.HvyAlcoholConsump,
                    p.BMI,
                    p.PhysActivity,
                    p.Smoker,
                ]
            ],
            columns=FEATURES,
        )
        scaled = scaler.transform(raw).astype("float32")

        # 3. model predict → probability
        prob = float(model(scaled, training=False).numpy()[0][0])
        pred = int(prob >= THRESHOLD)
        risk = label_risk(prob)

        # 4. Explainability: SHAP jika tersedia, fallback jika dependency belum siap.
        factors, explanation_method = explain_scaled_input(scaled)

        # 5. LLM → recommendation text
        patient_dict = p.model_dump()
        rec = generate_recommendation(patient_dict, prob, risk, factors)

        return PredictResponse(
            probability=round(prob, 4),
            risk_level=risk,
            prediction=pred,
            threshold_used=THRESHOLD,
            top_risk_factors=[FactorItem(**f) for f in factors],
            explanation_method=explanation_method,
            ai_recommendation=rec,
        )
    except ArtifactNotReadyError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
