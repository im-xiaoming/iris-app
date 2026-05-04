from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


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
    train_path = base_dir / config["data"]["processed_train_path"]
    test_path = base_dir / config["data"]["processed_test_path"]
    model_path = base_dir / config["training"]["model_path"]
    metrics_path = base_dir / config["training"]["metrics_path"]
    target_col = config["data"]["target_column"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    model = RandomForestClassifier(
        n_estimators=config["training"]["n_estimators"],
        max_depth=config["training"]["max_depth"],
        min_samples_split=config["training"]["min_samples_split"],
        min_samples_leaf=config["training"]["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_malignant": f1_score(y_test, predictions, pos_label="malignant"),
        "model_type": config["training"]["model_type"],
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }

    ensure_parent(model_path)
    ensure_parent(metrics_path)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training pipeline completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
