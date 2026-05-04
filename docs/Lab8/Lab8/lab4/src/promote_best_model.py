from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    winner_path = base_dir / config["artifacts"]["winner_path"]
    model_a_path = base_dir / config["artifacts"]["model_a_path"]
    model_b_path = base_dir / config["artifacts"]["model_b_path"]
    production_dir = base_dir / "deployment" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    winner_report = json.loads(winner_path.read_text(encoding="utf-8"))
    selected = winner_report["selected_best_model"]

    if selected == "model_a":
        source = model_a_path
    elif selected == "model_b":
        source = model_b_path
    else:
        raise SystemExit("Cannot promote model because A/B test result is a tie.")

    shutil.copy2(source, production_dir / "best_model.joblib")
    print(f"Promoted {selected} to production.")


if __name__ == "__main__":
    main()
