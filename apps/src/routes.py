import numpy as np
from fastapi import APIRouter, HTTPException

from src.schemas import PatientInput, PredictResponse, FactorItem
from src.services import model, scaler, explainer, FEATURES, THRESHOLD, label_risk
from src.explain import get_top_risk_factors
from src.llm import generate_recommendation

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status"   : "ok",
        "threshold": THRESHOLD,
        "params"   : int(model.count_params()),
    }


@router.get("/features")
def features():
    return {
        "features": FEATURES,
        "note"    : "Age_BMI_Risk dihitung otomatis dari Age x BMI",
        "fields"  : {
            "HighBP"           : "0/1",
            "GenHlth"          : "1-5",
            "HighChol"         : "0/1",
            "Age"              : "1-13 (grup usia CDC)",
            "CholCheck"        : "0/1",
            "HvyAlcoholConsump": "0/1",
            "BMI"              : "float",
            "PhysActivity"     : "0/1",
            "Smoker"           : "0/1",
        },
    }


@router.post("/predict", response_model=PredictResponse)
def predict(p: PatientInput):
    try:
        # 1. feature engineering 
        age_bmi = p.Age * p.BMI

        # 2. scaling
        raw    = np.array([[p.HighBP, p.GenHlth, p.HighChol, age_bmi,
                            p.CholCheck, p.HvyAlcoholConsump, p.BMI,
                            p.PhysActivity, p.Smoker]], dtype=np.float32)
        scaled = scaler.transform(raw).astype(np.float32)

        # 3. model predict → probability
        prob = float(model(scaled, training=False).numpy()[0][0])
        pred = int(prob >= THRESHOLD)
        risk = label_risk(prob)

        # 4. SHAP → top 3 faktor
        sv = explainer.shap_values(scaled)
        if isinstance(sv, list):
            sv = sv[0]
        factors = get_top_risk_factors(sv[0], FEATURES)

        # 5. LLM → recommendation text
        patient_dict = p.model_dump()
        rec = generate_recommendation(patient_dict, prob, risk, factors)

        return PredictResponse(
            probability      =round(prob, 4),
            risk_level       =risk,
            prediction       =pred,
            threshold_used   =THRESHOLD,
            top_risk_factors =[FactorItem(**f) for f in factors],
            ai_recommendation=rec,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
