from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "run_summary.csv"


def main() -> None:
    if not SUMMARY_PATH.exists():
        print("Chua co run_summary.csv. Hay chay train_experiments.py truoc.")
        return

    df = pd.read_csv(SUMMARY_PATH)
    df = df.sort_values(by=["f1", "accuracy", "roc_auc"], ascending=False)

    print("Bang so sanh model:")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print("\nMo hinh tot nhat hien tai:")
    print(f"- model_name: {best['model_name']}")
    print(f"- f1: {best['f1']}")
    print(f"- accuracy: {best['accuracy']}")
    print(f"- roc_auc: {best['roc_auc']}")


if __name__ == "__main__":
    main()
