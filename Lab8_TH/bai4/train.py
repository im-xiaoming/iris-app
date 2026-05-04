from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    out = Path("out")
    out.mkdir(exist_ok=True)

    x, y = make_classification(
        n_samples=1200,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        random_state=42,
    )

    tr_x, _, tr_y, _ = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    a = LogisticRegression(max_iter=400)
    b = DecisionTreeClassifier(max_depth=5, random_state=42)

    a.fit(tr_x, tr_y)
    b.fit(tr_x, tr_y)

    joblib.dump(a, out / "a.pkl")
    joblib.dump(b, out / "b.pkl")

    print("done")


if __name__ == "__main__":
    main()
