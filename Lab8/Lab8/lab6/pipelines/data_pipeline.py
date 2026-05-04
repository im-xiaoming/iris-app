from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = load_config()
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / config["data"]["raw_path"]
    train_path = base_dir / config["data"]["processed_train_path"]
    test_path = base_dir / config["data"]["processed_test_path"]
    target_col = config["data"]["target_column"]

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame.rename(columns={"target": target_col}, inplace=True)
    frame[target_col] = frame[target_col].map({0: "malignant", 1: "benign"})

    ensure_parent(raw_path)
    ensure_parent(train_path)
    ensure_parent(test_path)

    frame.to_csv(raw_path, index=False)

    train_df, test_df = train_test_split(
        frame,
        test_size=config["data"]["test_size"],
        random_state=seed,
        stratify=frame[target_col],
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Raw data saved to {raw_path.as_posix()}")
    print(f"Processed train data saved to {train_path.as_posix()}")
    print(f"Processed test data saved to {test_path.as_posix()}")


if __name__ == "__main__":
    main()
