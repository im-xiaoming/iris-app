from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "credit_default.csv"
DB_PATH = PROJECT_ROOT / "mlflow.db"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"


def tracking_uri() -> str:
    return f"sqlite:///{DB_PATH.as_posix()}"


def artifact_uri() -> str:
    return ARTIFACT_ROOT.resolve().as_uri()


def configure_mlflow() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["defaulted", "customer_id"])
    y = df["defaulted"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "income",
        "loan_amount",
        "age",
        "late_payments",
        "credit_utilization",
        "employment_years",
        "has_collateral",
    ]
    categorical_features = ["loan_purpose"]
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


def build_model(variant: str):
    if variant == "logistic":
        estimator = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        params = {"variant": variant, "C": 1.0, "random_state": 42}
    elif variant == "tree":
        estimator = DecisionTreeClassifier(max_depth=4, random_state=42)
        params = {"variant": variant, "max_depth": 4, "random_state": 42}
    else:
        raise ValueError(f"Variant khong hop le: {variant}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )
    return pipeline, params


def train_and_evaluate(variant: str):
    X_train, X_test, y_train, y_test = load_data()
    pipeline, params = build_model(variant)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    return pipeline, params, metrics
