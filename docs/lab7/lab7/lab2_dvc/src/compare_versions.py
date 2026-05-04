from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics_path = PROJECT_ROOT / "metrics.json"
    if not metrics_path.exists():
        print("Chua co metrics.json. Hay chay `dvc repro` truoc.")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("Baseline metrics hien tai:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")
    print("\nSo sanh version voi DVC:")
    print("- dvc metrics diff")
    print("- dvc params diff")


if __name__ == "__main__":
    main()
