from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_pipeline(config: dict, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        min_samples_split=config["model"]["min_samples_split"],
        min_samples_leaf=config["model"]["min_samples_leaf"],
        random_state=config["seed"],
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def track_environment(base_dir: Path, config: dict) -> dict:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "seed": config["seed"],
        "working_directory": str(base_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    config_path = (base_dir / args.config).resolve()
    config = load_config(config_path)

    set_global_seed(config["seed"])

    data_path = base_dir / config["data"]["output_path"]
    target_col = config["data"]["target_column"]

    artifacts = config["artifacts"]
    model_path = base_dir / artifacts["model_path"]
    metrics_path = base_dir / artifacts["metrics_path"]
    predictions_path = base_dir / artifacts["predictions_path"]
    environment_path = base_dir / artifacts["environment_path"]
    config_snapshot_path = base_dir / artifacts["config_snapshot_path"]

    for path in [model_path, metrics_path, predictions_path, environment_path, config_snapshot_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(data_path)
    X = frame.drop(columns=[target_col])
    y = frame[target_col]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=y,
    )

    pipeline = build_pipeline(config, numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, pos_label="malignant")
    report = classification_report(y_test, predictions, output_dict=True)

    joblib.dump(pipeline, model_path)

    prediction_frame = X_test.copy()
    prediction_frame["actual"] = y_test.values
    prediction_frame["predicted"] = predictions
    prediction_frame.to_csv(predictions_path, index=False)

    metrics = {
        "accuracy": accuracy,
        "f1_malignant": f1,
        "seed": config["seed"],
        "model_type": config["model"]["type"],
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    environment = track_environment(base_dir, config)
    environment_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")

    config_snapshot_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    print("Training finished successfully.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path.as_posix()}")
    print(f"Saved environment snapshot to {environment_path.as_posix()}")


if __name__ == "__main__":
    main()
