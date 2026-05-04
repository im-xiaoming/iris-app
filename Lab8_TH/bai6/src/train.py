import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


def main():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    x1 = tr.drop(columns=["y"])
    y1 = tr["y"]
    x2 = te.drop(columns=["y"])
    y2 = te["y"]

    md = RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42)
    md.fit(x1, y1)

    prd = md.predict(x2)
    acc = accuracy_score(y2, prd)
    rec = recall_score(y2, prd)

    Path("deploy/live").mkdir(parents=True, exist_ok=True)
    Path("monitor").mkdir(exist_ok=True)

    joblib.dump(md, "deploy/live/model.pkl")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fraud_e2e")
    with mlflow.start_run(run_name="fraud_rf"):
        mlflow.log_param("n", 120)
        mlflow.log_param("d", 7)
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("rec", rec)
        mlflow.sklearn.log_model(md, artifact_path="model")

    Path("monitor/train.json").write_text(
        json.dumps({"acc": round(float(acc), 4), "rec": round(float(rec), 4)}),
        encoding="utf-8",
    )

    print("done")
    print(f"acc {acc:.4f}")


if __name__ == "__main__":
    main()
