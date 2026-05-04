import json
import time
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


app = Flask(__name__)
mp = Path("deploy/app/model.pkl")
if not mp.exists():
    mp = Path("deploy/live/model.pkl")
md = joblib.load(mp)
log = Path("monitor/pred.jsonl")
log.parent.mkdir(parents=True, exist_ok=True)


def save(row):
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


@app.get("/")
def home():
    return {"msg": "bai6 ok", "ping": "/ping", "predict": "/predict", "report": "/report"}


@app.get("/ping")
def ping():
    return {"ok": 1}


@app.post("/predict")
def predict():
    d = request.get_json(force=True)
    st = time.perf_counter()
    y = int(md.predict([d["x"]])[0])
    ms = round((time.perf_counter() - st) * 1000, 3)
    save({"y": y, "ms": ms})
    return jsonify({"y": y, "ms": ms})


@app.get("/report")
def report():
    if not log.exists():
        return {"n": 0, "ms": 0}
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    n = len(rows)
    ms = round(sum(x["ms"] for x in rows) / n, 3) if n else 0
    return {"n": n, "ms": ms}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
