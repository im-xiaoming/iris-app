from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def print_top_trials(path: Path, title: str) -> None:
    if not path.exists():
        print(f"Chua co file {path.name}")
        return

    df = pd.read_csv(path).sort_values(by=["f1", "roc_auc"], ascending=False)
    print(f"\n{title}")
    print(df.head(5).to_string(index=False))


def main() -> None:
    print_top_trials(ARTIFACT_DIR / "trials_bayesian.csv", "Top trial Bayesian")
    print_top_trials(ARTIFACT_DIR / "trials_grid.csv", "Top trial Grid Search")


if __name__ == "__main__":
    main()
