import hashlib
import json
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


app = Flask(__name__)
out = Path("out")
log = Path("out/log.jsonl")
a = joblib.load(out / "a.pkl")
b = joblib.load(out / "b.pkl")


def pick(uid):
    v = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100
    return "a" if v < 50 else "b"


def save(row):
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load():
    if not log.exists():
        return []
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]


@app.get("/")
def home():
    return {"msg": "bai4 ok", "predict": "/predict", "stats": "/stats", "best": "/best"}


@app.post("/predict")
def predict():
    d = request.get_json(force=True)
    uid = str(d.get("id", "u"))
    x = [d["x"]]
    y = d.get("y")
    m = pick(uid)
    md = a if m == "a" else b
    prd = int(md.predict(x)[0])
    row = {"id": uid, "m": m, "p": prd}
    if y is not None:
        row["y"] = int(y)
        row["ok"] = int(int(y) == prd)
    save(row)
    return jsonify({"m": m, "y": prd})


@app.get("/stats")
def stats():
    rows = load()
    res = {"a": {"n": 0, "ok": 0}, "b": {"n": 0, "ok": 0}}
    for r in rows:
        k = r["m"]
        res[k]["n"] += 1
        res[k]["ok"] += r.get("ok", 0)
    for k in res:
        n = res[k]["n"]
        res[k]["acc"] = round(res[k]["ok"] / n, 4) if n else 0
    return jsonify(res)


@app.get("/best")
def best():
    d = stats().get_json()
    aa = d["a"]["acc"]
    bb = d["b"]["acc"]
    m = "a" if aa >= bb else "b"
    return jsonify({"best": m, "a": aa, "b": bb})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
