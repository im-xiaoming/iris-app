from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from common import PROJECT_ROOT, artifact_uri, configure_mlflow, train_and_evaluate


def ensure_experiment(experiment_name: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    return client.create_experiment(experiment_name, artifact_location=artifact_uri())


def wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout: int = 30) -> None:
    start = time.time()
    while time.time() - start < timeout:
        mv = client.get_model_version(model_name, version)
        if mv.status == "READY":
            return
        time.sleep(1)
    raise TimeoutError("Model version chua san sang sau thoi gian cho.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register a new model version in MLflow")
    parser.add_argument("--model-name", default="credit_default_model")
    parser.add_argument("--variant", choices=["logistic", "tree"], default="logistic")
    args = parser.parse_args()

    configure_mlflow()
    experiment_id = ensure_experiment("lab6_model_registry")
    client = MlflowClient()

    pipeline, params, metrics = train_and_evaluate(args.variant)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"train_{args.variant}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("lab", "lab6_model_registry")
        mlflow.set_tag("candidate", "true")
        artifact_path = "model"
        mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)

        summary_path = PROJECT_ROOT / "artifacts" / f"{args.variant}_metrics.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            "\n".join([f"{key}={value:.4f}" for key, value in metrics.items()]),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(summary_path))

        model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
        registered = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    wait_until_ready(client, args.model_name, registered.version)
    client.set_registered_model_tag(args.model_name, "domain", "credit_default")
    client.set_model_version_tag(args.model_name, registered.version, "variant", args.variant)
    client.set_model_version_tag(args.model_name, registered.version, "status", "candidate")
    print(f"Da dang ky model `{args.model_name}` version {registered.version}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
