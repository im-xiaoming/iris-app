from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_risk.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TRACKING_DIR = PROJECT_ROOT / "mlruns"


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


def build_pipeline(params: dict) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] in (-1, None) else int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=42,
        n_jobs=1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def evaluate_params(params: dict, X_train, X_test, y_train, y_test) -> tuple[dict, Pipeline]:
    pipeline = build_pipeline(params)
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="f1").mean()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_f1": float(cv_score),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }
    return metrics, pipeline


def run_grid_search(X_train, X_test, y_train, y_test) -> tuple[list[dict], dict, Pipeline]:
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
    }

    results = []
    best_metrics = None
    best_params = None
    best_model = None

    with mlflow.start_run(run_name="grid_search_rf") as parent_run:
        mlflow.set_tag("search_type", "grid")
        mlflow.log_param("parallel_jobs", 1)
        for index, params in enumerate(ParameterGrid(param_grid), start=1):
            with mlflow.start_run(run_name=f"grid_trial_{index}", nested=True):
                metrics, model = evaluate_params(params, X_train, X_test, y_train, y_test)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                result = {"trial_number": index, **params, **{k: round(v, 4) for k, v in metrics.items()}}
                results.append(result)

                if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                    best_metrics = metrics
                    best_params = params
                    best_model = model

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        mlflow.sklearn.log_model(best_model, artifact_path="best_model_grid")
        mlflow.set_tag("best_model_selected", "true")

    return results, best_params, best_model


def run_bayesian_search(X_train, X_test, y_train, y_test, trials: int, jobs: int) -> tuple[list[dict], dict, Pipeline]:
    trial_results: list[dict] = []
    best_holder: dict = {"params": None, "metrics": None, "model": None}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        }

        with mlflow.start_run(run_name=f"bayes_trial_{trial.number}", nested=True):
            metrics, model = evaluate_params(params, X_train, X_test, y_train, y_test)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("trial_number", trial.number)

        row = {"trial_number": trial.number, **params, **{k: round(v, 4) for k, v in metrics.items()}}
        trial_results.append(row)

        if best_holder["metrics"] is None or metrics["f1"] > best_holder["metrics"]["f1"]:
            best_holder["params"] = params
            best_holder["metrics"] = metrics
            best_holder["model"] = model

        return metrics["f1"]

    with mlflow.start_run(run_name="bayesian_search_rf"):
        mlflow.set_tag("search_type", "bayesian")
        mlflow.log_param("parallel_jobs", jobs)
        mlflow.log_param("n_trials", trials)
        study = optuna.create_study(direction="maximize", study_name="rf_bayesian_search")
        study.optimize(objective, n_trials=trials, n_jobs=jobs)
        mlflow.log_params({f"best_{k}": v for k, v in best_holder["params"].items()})
        mlflow.log_metrics({f"best_{k}": v for k, v in best_holder["metrics"].items()})
        mlflow.sklearn.log_model(best_holder["model"], artifact_path="best_model_bayesian")
        mlflow.set_tag("best_model_selected", "true")

    return trial_results, best_holder["params"], best_holder["model"]


def persist_results(search_type: str, results: list[dict], best_params: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).sort_values(by=["f1", "roc_auc"], ascending=False)
    results_df.to_csv(ARTIFACT_DIR / f"trials_{search_type}.csv", index=False)
    (ARTIFACT_DIR / f"best_params_{search_type}.json").write_text(
        json.dumps(best_params, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RandomForest model with Optuna or grid search")
    parser.add_argument("--search", choices=["grid", "bayesian"], default="bayesian")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("lab5_optuna_tuning")

    X_train, X_test, y_train, y_test = load_data()
    if args.search == "grid":
        results, best_params, _ = run_grid_search(X_train, X_test, y_train, y_test)
    else:
        results, best_params, _ = run_bayesian_search(X_train, X_test, y_train, y_test, args.trials, args.jobs)

    persist_results(args.search, results, best_params)
    print(f"Da hoan thanh tuning theo kieu: {args.search}")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
