from __future__ import annotations

import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = load_config()
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / config["data"]["raw_path"]
    model_path = base_dir / config["artifacts"]["model_path"]
    metrics_path = base_dir / config["artifacts"]["metrics_path"]

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame.rename(columns={"target": config["data"]["target_column"]}, inplace=True)
    frame[config["data"]["target_column"]] = frame[config["data"]["target_column"]].map(
        {0: "malignant", 1: "benign"}
    )

    ensure_parent(raw_path)
    frame.to_csv(raw_path, index=False)

    target_col = config["data"]["target_column"]
    X = frame.drop(columns=[target_col])
    y = frame[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_malignant": f1_score(y_test, predictions, pos_label="malignant"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    ensure_parent(model_path)
    ensure_parent(metrics_path)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
