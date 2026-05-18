from pydantic import BaseModel, Field
from typing import List


class PatientInput(BaseModel):
    HighBP            : int   = Field(..., ge=0, le=1)
    GenHlth           : int   = Field(..., ge=1, le=5)
    HighChol          : int   = Field(..., ge=0, le=1)
    Age               : int   = Field(..., ge=1, le=13)
    CholCheck         : int   = Field(..., ge=0, le=1)
    HvyAlcoholConsump : int   = Field(..., ge=0, le=1)
    BMI               : float = Field(..., ge=10.0, le=100.0)
    PhysActivity      : int   = Field(..., ge=0, le=1)
    Smoker            : int   = Field(..., ge=0, le=1)


class FactorItem(BaseModel):
    feature    : str
    shap_value : float
    direction  : str


class PredictResponse(BaseModel):
    probability      : float
    risk_level       : str
    prediction       : int
    threshold_used   : float
    top_risk_factors : List[FactorItem]
    ai_recommendation: str
