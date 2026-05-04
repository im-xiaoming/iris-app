from __future__ import annotations

from pathlib import Path

from kfp import dsl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_IMAGE = "python:3.10"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["pandas==2.2.0"])
def trigger_training(dataset_path: str) -> str:
    from datetime import datetime
    from pathlib import Path

    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {dataset}")

    run_id = f"kubeflow_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return run_id


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.0", "scikit-learn==1.4.0"],
)
def preprocess_data(
    dataset_path: str,
    train_output_path: str,
    test_output_path: str,
) -> str:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset_path)
    df["internet_access"] = (df["internet_access"] == "yes").astype(int)
    df["parent_support"] = df["parent_support"].map({"low": 0, "medium": 1, "high": 2})

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["passed"],
    )

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    return f"train_rows={len(train_df)}, test_rows={len(test_df)}"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.0", "scikit-learn==1.4.0", "joblib==1.3.2"],
)
def train_model(train_data_path: str, model_output_path: str) -> str:
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop(columns=["passed", "student_id"])
    y_train = train_df["passed"]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_output_path)
    return "model_trained"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.0", "scikit-learn==1.4.0", "joblib==1.3.2"],
)
def evaluate_model(test_data_path: str, model_path: str, metrics_output_path: str) -> str:
    import json

    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    test_df = pd.read_csv(test_data_path)
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

    with open(metrics_output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return json.dumps(metrics)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["joblib==1.3.2"])
def save_model(run_id: str, model_path: str, metrics_json: str, save_summary_path: str) -> str:
    import json
    from pathlib import Path

    target = Path(save_summary_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "model_path": model_path,
        "metrics": json.loads(metrics_json),
        "status": "saved",
    }
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(target)


@dsl.pipeline(name="automated-training-pipeline")
def automated_training_pipeline(
    dataset_path: str = str(PROJECT_ROOT / "data" / "student_performance.csv"),
    train_output_path: str = str(PROJECT_ROOT / "artifacts" / "train.csv"),
    test_output_path: str = str(PROJECT_ROOT / "artifacts" / "test.csv"),
    model_output_path: str = str(PROJECT_ROOT / "artifacts" / "model.pkl"),
    metrics_output_path: str = str(PROJECT_ROOT / "artifacts" / "evaluation.json"),
    save_summary_path: str = str(PROJECT_ROOT / "artifacts" / "save_summary.json"),
):
    trigger_task = trigger_training(dataset_path=dataset_path)
    preprocess_task = preprocess_data(
        dataset_path=dataset_path,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
    )
    train_task = train_model(
        train_data_path=train_output_path,
        model_output_path=model_output_path,
    )
    evaluate_task = evaluate_model(
        test_data_path=test_output_path,
        model_path=model_output_path,
        metrics_output_path=metrics_output_path,
    )
    save_task = save_model(
        run_id=trigger_task.output,
        model_path=model_output_path,
        metrics_json=evaluate_task.output,
        save_summary_path=save_summary_path,
    )

    preprocess_task.after(trigger_task)
    train_task.after(preprocess_task)
    evaluate_task.after(train_task)
    save_task.after(evaluate_task)
