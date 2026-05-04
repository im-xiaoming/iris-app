from pathlib import Path

import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def fe(df):
    df = df.copy()
    df["f1"] = df["mean radius"] * df["mean texture"]
    df["f2"] = df["mean area"] / (df["mean perimeter"] + 1e-6)
    df["f3"] = df["worst radius"] - df["mean radius"]
    return df


def main():
    out = Path("out")
    out.mkdir(exist_ok=True)

    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.rename(columns={"target": "y"})
    df = fe(df)

    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y"])

    h2o.init()
    h2o.remove_all()

    htr = h2o.H2OFrame(tr)
    hte = h2o.H2OFrame(te)
    htr["y"] = htr["y"].asfactor()
    hte["y"] = hte["y"].asfactor()

    x = [c for c in htr.columns if c != "y"]
    aml = H2OAutoML(max_models=6, seed=42, sort_metric="AUC")
    aml.train(x=x, y="y", training_frame=htr)

    prd = aml.leader.predict(hte).as_data_frame()
    acc = accuracy_score(te["y"], prd["predict"].astype(int))
    auc = roc_auc_score(te["y"], prd["p1"])

    aml.leaderboard.as_data_frame().to_csv(out / "board.csv", index=False)
    p = h2o.save_model(aml.leader, path=str(out), force=True)

    print("done")
    print(f"acc {acc:.4f}")
    print(f"auc {auc:.4f}")
    print(p)

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
