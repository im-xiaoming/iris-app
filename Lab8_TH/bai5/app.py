import json
import time
from pathlib import Path

import joblib
import redis
from fastapi import FastAPI
from pydantic import BaseModel


class Inp(BaseModel):
    x: list[float]


app = FastAPI()
out = Path("out")
out.mkdir(exist_ok=True)
log = out / "app.log"
md = joblib.load(out / "model.pkl")
mem = {}

try:
    rd = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    rd.ping()
except Exception:
    rd = None


def key(x):
    return ",".join(str(round(v, 6)) for v in x)


def pull(k):
    if rd is not None:
        v = rd.get(k)
        if v is not None:
            return int(v), 1
    if k in mem:
        return mem[k], 1
    return None, 0


def push(k, v):
    if rd is not None:
        rd.setex(k, 60, int(v))
    mem[k] = int(v)


def save(row):
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


@app.get("/")
def home():
    return {"msg": "bai5 ok", "ping": "/ping", "predict": "/predict", "docs": "/docs"}


@app.get("/ping")
def ping():
    return {"ok": 1}


@app.post("/predict")
def predict(inp: Inp):
    st = time.perf_counter()
    k = key(inp.x)
    y, hit = pull(k)
    if y is None:
        y = int(md.predict([inp.x])[0])
        push(k, y)
    ms = round((time.perf_counter() - st) * 1000, 3)
    save({"x": inp.x, "y": y, "hit": hit, "ms": ms})
    return {"y": y, "hit": hit, "ms": ms}
