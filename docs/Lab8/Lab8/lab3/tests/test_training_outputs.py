from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def test_metrics_file_exists() -> None:
    metrics_path = BASE_DIR / "reports" / "metrics.json"
    assert metrics_path.exists(), "metrics.json must exist after training"


def test_metrics_quality_threshold() -> None:
    metrics_path = BASE_DIR / "reports" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["accuracy"] >= 0.93
    assert metrics["f1_malignant"] >= 0.92
