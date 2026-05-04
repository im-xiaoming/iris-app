from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse


BASE_DIR = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    config_path = BASE_DIR / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()
model = joblib.load(BASE_DIR / config["deployment"]["deployed_model_path"])
request_log_path = BASE_DIR / config["monitoring"]["request_log_path"]
request_log_path.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="End-to-End MLOps Service", version="1.0.0")

REQUEST_COUNT = Counter("prediction_requests_total", "Total number of prediction requests")
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Latency for prediction requests")


class PredictionRequest(BaseModel):
    request_id: str = Field(..., description="Unique identifier")
    features: list[float] = Field(..., min_length=30, max_length=30)


def append_prediction_log(payload: dict) -> None:
    with request_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return generate_latest().decode("utf-8")


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    REQUEST_COUNT.inc()
    started_at = time.perf_counter()

    try:
        prediction = model.predict([payload.features])[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid payload for prediction") from exc

    latency = time.perf_counter() - started_at
    REQUEST_LATENCY.observe(latency)

    log_record = {
        "request_id": payload.request_id,
        "prediction": prediction,
        "latency_ms": round(latency * 1000, 3),
        "timestamp": time.time(),
    }
    append_prediction_log(log_record)

    return log_record
