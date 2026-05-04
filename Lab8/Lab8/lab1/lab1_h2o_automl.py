from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import h2o
    from h2o.automl import H2OAutoML
except ModuleNotFoundError as exc:  # noqa: BLE001
    raise SystemExit(
        "Missing dependency 'h2o'. Install requirements with "
        "'py -3.10 -m pip install -r requirements.txt' and run again."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "breast_cancer_with_missing.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.json"
LEADERBOARD_PATH = OUTPUT_DIR / "leaderboard.csv"


def auto_feature_engineering(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    engineered = frame.copy()
    feature_cols = [col for col in engineered.columns if col != target_col]

    numeric_cols = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(engineered[col])
    ]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in numeric_cols:
        if engineered[col].isna().any():
            engineered[f"{col}_missing"] = engineered[col].isna().astype("category")
        engineered[col] = engineered[col].fillna(engineered[col].median())

    for col in categorical_cols:
        engineered[col] = engineered[col].fillna("Unknown").astype(str)
        top_categories = engineered[col].value_counts().nlargest(5).index
        engineered[f"{col}_top5_flag"] = engineered[col].isin(top_categories).astype(
            "category"
        )

    top_numeric = sorted(
        numeric_cols,
        key=lambda column: engineered[column].var(),
        reverse=True,
    )[:4]

    for idx in range(len(top_numeric)):
        for jdx in range(idx + 1, len(top_numeric)):
            left = top_numeric[idx]
            right = top_numeric[jdx]
            engineered[f"{left}_x_{right}"] = engineered[left] * engineered[right]

    for col in top_numeric[:3]:
        ranked = engineered[col].rank(method="first")
        engineered[f"{col}_bin"] = pd.qcut(
            ranked,
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        ).astype(str)

    engineered[target_col] = engineered[target_col].astype(str)
    return engineered


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def export_text_leaderboard(leaderboard_frame, output_path: Path) -> None:
    leaderboard_pd = h2o.as_list(leaderboard_frame, use_pandas=True)
    leaderboard_pd.to_csv(output_path, index=False)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run generate_dataset.py first."
        )

    ensure_directories([OUTPUT_DIR, MODEL_DIR])

    source_df = pd.read_csv(DATA_PATH)
    target_col = "diagnosis"
    engineered_df = auto_feature_engineering(source_df, target_col=target_col)
    engineered_data_path = OUTPUT_DIR / "engineered_dataset.csv"
    engineered_df.to_csv(engineered_data_path, index=False)

    h2o.init(max_mem_size="2G")
    h2o.no_progress()

    try:
        hf = h2o.H2OFrame(engineered_df)
        hf[target_col] = hf[target_col].asfactor()

        train, valid, test = hf.split_frame(ratios=[0.7, 0.15], seed=42)
        x = [col for col in hf.columns if col != target_col]

        automl = H2OAutoML(
            max_models=12,
            seed=42,
            sort_metric="AUC",
            balance_classes=True,
            include_algos=["GBM", "XGBoost", "DRF", "GLM", "StackedEnsemble"],
        )
        automl.train(x=x, y=target_col, training_frame=train, validation_frame=valid)

        leader = automl.leader
        perf = leader.model_performance(test_data=test)
        leaderboard = automl.leaderboard

        export_text_leaderboard(leaderboard, LEADERBOARD_PATH)
        model_path = h2o.save_model(model=leader, path=str(MODEL_DIR), force=True)

        mojo_path = None
        try:
            mojo_path = h2o.download_mojo(
                model=leader,
                path=str(MODEL_DIR),
                get_genmodel_jar=False,
            )
        except Exception as exc:  # noqa: BLE001
            mojo_path = f"MOJO export skipped: {exc}"

        report = {
            "dataset_path": str(DATA_PATH.resolve()),
            "engineered_dataset_path": str(engineered_data_path.resolve()),
            "target_column": target_col,
            "feature_count_after_engineering": len(x),
            "leader_model_id": leader.model_id,
            "leader_algorithm": leader.algo,
            "metrics": {
                "auc": perf.auc(),
                "accuracy": perf.accuracy()[0][1],
                "f1": perf.F1()[0][1],
                "logloss": perf.logloss(),
            },
            "exported_model_path": str(Path(model_path).resolve()),
            "exported_mojo_path": str(mojo_path),
            "leaderboard_path": str(LEADERBOARD_PATH.resolve()),
        }

        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== AUTO FEATURE ENGINEERING COMPLETE ===")
        print(f"Engineered dataset saved to: {engineered_data_path.resolve()}")
        print(f"Feature count after engineering: {len(x)}")

        print("\n=== MODEL SELECTION COMPLETE ===")
        print(f"Leader model id: {leader.model_id}")
        print(f"Leader algorithm: {leader.algo}")
        print(h2o.as_list(leaderboard.head(rows=10), use_pandas=True))

        print("\n=== EVALUATION COMPLETE ===")
        print(json.dumps(report["metrics"], indent=2))

        print("\n=== EXPORT COMPLETE ===")
        print(f"Binary model path: {Path(model_path).resolve()}")
        print(f"MOJO path/status: {mojo_path}")
        print(f"Report path: {REPORT_PATH.resolve()}")
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
