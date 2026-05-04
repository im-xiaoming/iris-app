from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def main():
    raw = Path("data")
    raw.mkdir(exist_ok=True)

    x, y = make_classification(
        n_samples=3000,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        weights=[0.95, 0.05],
        random_state=42,
    )

    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(12)])
    df["y"] = y

    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])

    df.to_csv(raw / "fraud.csv", index=False)
    tr.to_csv(raw / "train.csv", index=False)
    te.to_csv(raw / "test.csv", index=False)

    print("done")


if __name__ == "__main__":
    main()
