from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def trigger_training() -> str:
    run_id = f"local_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Trigger training: {run_id}")
    return run_id


def preprocess_data() -> tuple[Path, Path]:
    df = pd.read_csv(DATA_PATH)
    df["internet_access"] = (df["internet_access"] == "yes").astype(int)
    df["parent_support"] = df["parent_support"].map({"low": 0, "medium": 1, "high": 2})

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["passed"],
    )

    train_path = ARTIFACT_DIR / "train.csv"
    test_path = ARTIFACT_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Preprocessing xong: {train_path}, {test_path}")
    return train_path, test_path


def train_model(train_path: Path) -> Path:
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["passed", "student_id"])
    y_train = train_df["passed"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    model_path = ARTIFACT_DIR / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Train xong model: {model_path}")
    return model_path


def evaluate_model(test_path: Path, model_path: Path) -> Path:
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["passed", "student_id"])
    y_test = test_df["passed"]

    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }

    metrics_path = ARTIFACT_DIR / "evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Evaluation xong: {metrics}")
    return metrics_path


def save_model(run_id: str, model_path: Path, metrics_path: Path) -> Path:
    summary = {
        "run_id": run_id,
        "model_path": str(model_path),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "status": "saved",
    }

    summary_path = ARTIFACT_DIR / "save_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Luu model metadata xong: {summary_path}")
    return summary_path


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = trigger_training()
    train_path, test_path = preprocess_data()
    model_path = train_model(train_path)
    metrics_path = evaluate_model(test_path, model_path)
    save_model(run_id, model_path, metrics_path)
    print("Automated training pipeline da hoan thanh.")


if __name__ == "__main__":
    main()
