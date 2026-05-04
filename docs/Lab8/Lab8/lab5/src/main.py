from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_app.docs.Lab8.Lab8.lab5.src.cache import RedisCache
from ml_app.docs.Lab8.Lab8.lab5.src.logger_config import setup_logger


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "params.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


config = load_config()
logger = setup_logger(BASE_DIR / config["logging"]["log_path"])
model = joblib.load(BASE_DIR / config["api"]["model_path"])
cache = RedisCache(
    host=config["redis"]["host"],
    port=config["redis"]["port"],
    db=config["redis"]["db"],
    ttl_seconds=config["redis"]["ttl_seconds"],
    prefix=config["redis"]["prediction_prefix"],
)

app = FastAPI(title="Real-time Prediction System", version="1.0.0")


class PredictionRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    features: list[float] = Field(..., min_length=30, max_length=30)


def fingerprint_features(features: list[float]) -> str:
    payload = json.dumps(features, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    started_at = time.perf_counter()
    feature_hash = fingerprint_features(payload.features)

    try:
        cached_response = cache.get(feature_hash)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cache_read_failed request_id=%s error=%s",
            payload.request_id,
            str(exc),
        )
        cached_response = None

    if cached_response is not None:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
        logger.info(
            "prediction request_id=%s source=cache latency_ms=%s prediction=%s",
            payload.request_id,
            latency_ms,
            cached_response["prediction"],
        )
        return {
            **cached_response,
            "cache_hit": True,
            "latency_ms": latency_ms,
        }

    try:
        prediction = model.predict([payload.features])[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction_failed request_id=%s error=%s", payload.request_id, str(exc))
        raise HTTPException(status_code=400, detail="Invalid prediction payload") from exc

    response = {
        "request_id": payload.request_id,
        "prediction": prediction,
    }

    try:
        cache.set(feature_hash, response)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cache_write_failed request_id=%s error=%s",
            payload.request_id,
            str(exc),
        )

    latency_ms = round((time.perf_counter() - started_at) * 1000, 3)
    logger.info(
        "prediction request_id=%s source=model latency_ms=%s prediction=%s",
        payload.request_id,
        latency_ms,
        prediction,
    )

    return {
        **response,
        "cache_hit": False,
        "latency_ms": latency_ms,
    }
