from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TRACKING_DIR = PROJECT_ROOT / "mlruns"


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "support_calls",
        "is_premium",
        "late_payments",
    ]
    categorical_features = ["contract_type"]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_experiments() -> list[tuple[str, object, dict]]:
    return [
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            {"random_state": 42, "C": 1.0, "test_size": 0.3},
        ),
        (
            "decision_tree_depth3",
            DecisionTreeClassifier(max_depth=3, random_state=42),
            {"random_state": 42, "max_depth": 3, "test_size": 0.3},
        ),
        (
            "decision_tree_depth5",
            DecisionTreeClassifier(max_depth=5, random_state=42),
            {"random_state": 42, "max_depth": 5, "test_size": 0.3},
        ),
    ]


def evaluate_pipeline(model_name: str, estimator, params: dict, X_train, X_test, y_train, y_test) -> dict:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        summary = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    **params,
                    **{key: round(value, 4) for key, value in metrics.items()},
                }
            ]
        )
        summary_path = ARTIFACT_DIR / f"{model_name}_summary.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))

        run_id = mlflow.active_run().info.run_id

    return {
        "run_id": run_id,
        "model_name": model_name,
        **params,
        **{key: round(value, 4) for key, value in metrics.items()},
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("customer_churn_experiments")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    all_results = []
    for model_name, estimator, params in build_experiments():
        result = evaluate_pipeline(model_name, estimator, params, X_train, X_test, y_train, y_test)
        all_results.append(result)

    results_df = pd.DataFrame(all_results).sort_values(by="f1", ascending=False)
    results_path = ARTIFACT_DIR / "run_summary.csv"
    results_df.to_csv(results_path, index=False)
    print("Da hoan tat cac run MLflow.")
    print(results_df.to_string(index=False))
    print(f"\nTong hop ket qua: {results_path}")


if __name__ == "__main__":
    main()
