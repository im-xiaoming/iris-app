from pathlib import Path

import joblib
from flask import Flask, jsonify, request


app = Flask(__name__)
p = Path("deploy/current/model.pkl")
if not p.exists():
    p = Path("out/model.pkl")
md = joblib.load(p)


@app.get("/")
def home():
    return {"msg": "bai3 ok", "ping": "/ping", "predict": "/predict"}


@app.get("/ping")
def ping():
    return {"ok": 1}


@app.post("/predict")
def predict():
    d = request.get_json(force=True)
    x = [d["x"]]
    y = int(md.predict(x)[0])
    return jsonify({"y": y})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
