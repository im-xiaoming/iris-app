from __future__ import annotations

import argparse

from mlflow import MlflowClient

from common import configure_mlflow


def parse_tag(raw_tag: str) -> tuple[str, str]:
    if "=" not in raw_tag:
        raise ValueError("Tag phai co dang key=value")
    key, value = raw_tag.split("=", 1)
    return key, value


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model version by alias and tags")
    parser.add_argument("--model-name", default="credit_default_model")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    configure_mlflow()
    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    client.set_model_version_tag(args.model_name, args.version, "status", "promoted")

    for raw_tag in args.tag:
        key, value = parse_tag(raw_tag)
        client.set_model_version_tag(args.model_name, args.version, key, value)

    print(f"Da gan alias `{args.alias}` cho model {args.model_name} version {args.version}")


if __name__ == "__main__":
    main()
