from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    model_a_path = base_dir / config["artifacts"]["model_a_path"]
    model_b_path = base_dir / config["artifacts"]["model_b_path"]
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
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    model_a = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=seed,
        n_jobs=-1,
    )
    model_b = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=3,
        random_state=seed,
    )

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)

    metrics = {
        "model_a": {
            "name": "RandomForest",
            "accuracy": accuracy_score(y_test, pred_a),
            "f1_malignant": f1_score(y_test, pred_a, pos_label="malignant"),
        },
        "model_b": {
            "name": "GradientBoosting",
            "accuracy": accuracy_score(y_test, pred_b),
            "f1_malignant": f1_score(y_test, pred_b, pos_label="malignant"),
        },
    }

    ensure_parent(model_a_path)
    ensure_parent(model_b_path)
    ensure_parent(metrics_path)

    joblib.dump(model_a, model_a_path)
    joblib.dump(model_b, model_b_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Two models trained successfully.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
