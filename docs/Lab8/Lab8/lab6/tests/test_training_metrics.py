from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def test_training_metrics_exist() -> None:
    metrics_path = BASE_DIR / "artifacts" / "training_metrics.json"
    assert metrics_path.exists(), "training_metrics.json must exist after training"


def test_training_metrics_threshold() -> None:
    metrics_path = BASE_DIR / "artifacts" / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["accuracy"] >= 0.93
    assert metrics["f1_malignant"] >= 0.92
