import os
import sys
from pathlib import Path

os.environ["OPENROUTER_API_KEY"] = ""

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import Response

from apps.src.routes import features, health, predict
from apps.src.schemas import PatientInput


def main():
    health_response = Response()
    health_payload = health(health_response)
    print(
        "health:",
        health_payload["status"],
        f"ready={health_payload['ready']}",
        f"explainability_ready={health_payload['explainability_ready']}",
    )
    if health_response.status_code >= 400 or not health_payload["ready"]:
        raise SystemExit(f"core inference is not ready: {health_payload['artifacts']}")

    feature_payload = features()
    print("features:", len(feature_payload["features"]))

    sample = PatientInput(
        HighBP=1,
        GenHlth=3,
        HighChol=1,
        Age=8,
        CholCheck=1,
        HvyAlcoholConsump=0,
        BMI=28.5,
        PhysActivity=1,
        Smoker=0,
    )
    prediction = predict(sample)
    print(
        "predict:",
        f"probability={prediction.probability}",
        f"risk={prediction.risk_level}",
        f"method={prediction.explanation_method}",
    )


if __name__ == "__main__":
    main()
