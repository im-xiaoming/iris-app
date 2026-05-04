import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    out = Path("out")
    out.mkdir(exist_ok=True)

    ds = load_iris()
    x = ds.data
    y = ds.target

    tr_x, te_x, tr_y, te_y = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    md = LogisticRegression(max_iter=400)
    md.fit(tr_x, tr_y)

    acc = accuracy_score(te_y, md.predict(te_x))

    joblib.dump(md, out / "model.pkl")
    (out / "metrics.json").write_text(json.dumps({"acc": round(float(acc), 4)}), encoding="utf-8")

    print("done")
    print(f"acc {acc:.4f}")


if __name__ == "__main__":
    main()
