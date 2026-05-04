from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    request_log_path = base_dir / config["monitoring"]["request_log_path"]
    metrics_snapshot_path = base_dir / config["monitoring"]["metrics_snapshot_path"]
    drift_report_path = base_dir / config["monitoring"]["drift_report_path"]

    records = []
    if request_log_path.exists():
        with request_log_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if records:
        frame = pd.DataFrame(records)
        avg_latency_ms = float(frame["latency_ms"].mean())
        request_count = int(len(frame))
        prediction_distribution = frame["prediction"].value_counts().to_dict()
    else:
        avg_latency_ms = 0.0
        request_count = 0
        prediction_distribution = {}

    metrics_snapshot = {
        "request_count": request_count,
        "avg_latency_ms": avg_latency_ms,
        "prediction_distribution": prediction_distribution,
    }

    drift_report = {
        "status": "baseline",
        "note": "Simple placeholder drift report. Extend with PSI, KS test, or feature drift checks.",
        "request_count_evaluated": request_count,
    }

    metrics_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    drift_report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_snapshot_path.write_text(json.dumps(metrics_snapshot, indent=2), encoding="utf-8")
    drift_report_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")

    print(json.dumps(metrics_snapshot, indent=2))
    print(json.dumps(drift_report, indent=2))


if __name__ == "__main__":
    main()
