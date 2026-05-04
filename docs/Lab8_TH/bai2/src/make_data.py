from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine


def main():
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    ds = load_wine(as_frame=True)
    df = ds.frame.rename(columns={"target": "y"})
    df.to_csv(out / "wine.csv", index=False)

    print("done")


if __name__ == "__main__":
    main()
