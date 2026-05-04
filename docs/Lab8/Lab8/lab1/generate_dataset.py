from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    output_dir = BASE_DIR / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "breast_cancer_with_missing.csv"

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame.rename(columns={"target": "diagnosis"}, inplace=True)
    frame["diagnosis"] = frame["diagnosis"].map({0: "malignant", 1: "benign"})

    rng = np.random.default_rng(42)
    feature_cols = [col for col in frame.columns if col != "diagnosis"]

    for col in feature_cols[:8]:
        mask = rng.random(len(frame)) < 0.04
        frame.loc[mask, col] = np.nan

    frame.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path.as_posix()}")
    print(frame.head())


if __name__ == "__main__":
    main()
