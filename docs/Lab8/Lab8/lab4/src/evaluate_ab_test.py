from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_variant(frame: pd.DataFrame, variant: str) -> dict:
    subset = frame[frame["assigned_model"] == variant].dropna(subset=["actual_label"])
    if subset.empty:
        return {
            "samples": 0,
            "accuracy": None,
            "f1_malignant": None,
        }

    return {
        "samples": int(len(subset)),
        "accuracy": accuracy_score(subset["actual_label"], subset["prediction"]),
        "f1_malignant": f1_score(
            subset["actual_label"],
            subset["prediction"],
            pos_label="malignant",
        ),
    }


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    evaluation_path = base_dir / config["data"]["evaluation_path"]
    winner_path = base_dir / config["artifacts"]["winner_path"]

    frame = pd.read_csv(evaluation_path)
    result = {
        "model_a": evaluate_variant(frame, "model_a"),
        "model_b": evaluate_variant(frame, "model_b"),
    }

    a_score = result["model_a"]["f1_malignant"] or -1
    b_score = result["model_b"]["f1_malignant"] or -1

    if a_score > b_score:
        winner = "model_a"
    elif b_score > a_score:
        winner = "model_b"
    else:
        winner = "tie"

    final_report = {
        "comparison": result,
        "selected_best_model": winner,
    }

    winner_path.parent.mkdir(parents=True, exist_ok=True)
    winner_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
