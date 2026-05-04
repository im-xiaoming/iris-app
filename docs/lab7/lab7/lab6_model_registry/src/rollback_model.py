from __future__ import annotations

import argparse

from mlflow import MlflowClient

from ml_app.docs.lab7.lab7.lab6_model_registry.src.common import configure_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback alias to an older model version")
    parser.add_argument("--model-name", default="credit_default_model")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    configure_mlflow()
    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    client.set_model_version_tag(args.model_name, args.version, "rollback", "true")
    client.set_model_version_tag(args.model_name, args.version, "status", "active_after_rollback")
    print(f"Da rollback alias `{args.alias}` ve version {args.version} cua model {args.model_name}")


if __name__ == "__main__":
    main()
