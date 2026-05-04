from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
    raw_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "titanic.csv")

    df = raw_df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(params["prepare"]["fill_embarked_value"])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsFemale"] = (df["Sex"] == "female").astype(int)
    df = df.drop(columns=["Name", "Sex"])

    train_df, test_df = train_test_split(
        df,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"],
        stratify=df["Survived"],
    )

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    stats = {
        "rows_raw": len(raw_df),
        "rows_train": len(train_df),
        "rows_test": len(test_df),
    }
    (PROJECT_ROOT / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
