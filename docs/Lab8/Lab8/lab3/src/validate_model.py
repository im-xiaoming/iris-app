from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    metrics_path = base_dir / config["artifacts"]["metrics_path"]
    validation_path = base_dir / config["artifacts"]["validation_path"]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    accuracy_threshold = 0.93
    f1_threshold = 0.92

    passed = (
        metrics["accuracy"] >= accuracy_threshold
        and metrics["f1_malignant"] >= f1_threshold
    )

    result = {
        "passed": passed,
        "accuracy": metrics["accuracy"],
        "f1_malignant": metrics["f1_malignant"],
        "accuracy_threshold": accuracy_threshold,
        "f1_threshold": f1_threshold,
    }

    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not passed:
        raise SystemExit(
            "Model validation failed. Accuracy or F1 is below the deployment threshold."
        )

    print("Validation passed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
