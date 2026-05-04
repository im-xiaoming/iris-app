from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    config_path = (base_dir / args.config).resolve()
    config = load_config(config_path)

    output_path = base_dir / config["data"]["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame.rename(columns={"target": config["data"]["target_column"]}, inplace=True)
    frame[config["data"]["target_column"]] = frame[config["data"]["target_column"]].map(
        {0: "malignant", 1: "benign"}
    )

    frame.to_csv(output_path, index=False)
    print(f"Prepared data at {output_path.as_posix()}")
    print(frame.head())


if __name__ == "__main__":
    main()
