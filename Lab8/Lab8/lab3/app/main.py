from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

app = FastAPI(title="Lab 3 ML Service", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float]


def load_model():
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    model = load_model()
    prediction = model.predict([payload.features])[0]
    return {"prediction": prediction}
