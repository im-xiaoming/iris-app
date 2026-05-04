from __future__ import annotations

import argparse

from mlflow import MlflowClient

from common import configure_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description="List registered model versions")
    parser.add_argument("--model-name", default="credit_default_model")
    args = parser.parse_args()

    configure_mlflow()
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{args.model_name}'")

    if not versions:
        print("Chua co model version nao.")
        return

    print(f"Cac version cua model {args.model_name}:")
    for mv in sorted(versions, key=lambda item: int(item.version)):
        print(
            f"- version={mv.version} current_stage={mv.current_stage} "
            f"aliases={list(mv.aliases)} tags={dict(mv.tags)} run_id={mv.run_id}"
        )


if __name__ == "__main__":
    main()
