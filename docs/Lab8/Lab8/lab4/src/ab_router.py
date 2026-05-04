from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "params.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(model_path: Path):
    return joblib.load(model_path)


config = load_config()
MODEL_A = load_model(BASE_DIR / config["artifacts"]["model_a_path"])
MODEL_B = load_model(BASE_DIR / config["artifacts"]["model_b_path"])
EVALUATION_PATH = BASE_DIR / config["data"]["evaluation_path"]
EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def route_user(user_id: str) -> str:
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < config["traffic"]["model_a_weight"]:
        return "model_a"
    return "model_b"


def append_log(row: dict) -> None:
    frame = pd.DataFrame([row])
    header = not EVALUATION_PATH.exists()
    frame.to_csv(EVALUATION_PATH, mode="a", index=False, header=header)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    user_id = str(payload["user_id"])
    features = payload["features"]

    assigned_model = route_user(user_id)
    model = MODEL_A if assigned_model == "model_a" else MODEL_B
    prediction = model.predict([features])[0]

    log_row = {
        "user_id": user_id,
        "assigned_model": assigned_model,
        "prediction": prediction,
        "actual_label": payload.get("actual_label"),
    }
    append_log(log_row)

    return jsonify(
        {
            "user_id": user_id,
            "assigned_model": assigned_model,
            "prediction": prediction,
        }
    )


@app.get("/config")
def get_config():
    return jsonify(config)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
