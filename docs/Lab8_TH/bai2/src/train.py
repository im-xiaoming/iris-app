import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    cfg = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    out = Path("out")
    out.mkdir(exist_ok=True)

    df = pd.read_csv("data/raw/wine.csv")
    x = df.drop(columns=["y"])
    y = df["y"]

    tr_x, te_x, tr_y, te_y = train_test_split(
        x,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
        stratify=y,
    )

    md = RandomForestClassifier(
        n_estimators=cfg["model"]["n"],
        max_depth=cfg["model"]["d"],
        random_state=cfg["seed"],
    )
    md.fit(tr_x, tr_y)

    prd = md.predict(te_x)
    acc = accuracy_score(te_y, prd)

    joblib.dump(md, out / "model.pkl")
    shutil.copyfile("params.yaml", out / "params_used.yaml")

    txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (out / "env.txt").write_text(txt, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"acc": round(float(acc), 4)}), encoding="utf-8")

    print("done")
    print(f"acc {acc:.4f}")


if __name__ == "__main__":
    main()
