from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def main():
    out = Path("out")
    out.mkdir(exist_ok=True)

    ds = load_iris()
    md = RandomForestClassifier(n_estimators=80, random_state=42)
    md.fit(ds.data, ds.target)
    joblib.dump(md, out / "model.pkl")

    print("done")


if __name__ == "__main__":
    main()
