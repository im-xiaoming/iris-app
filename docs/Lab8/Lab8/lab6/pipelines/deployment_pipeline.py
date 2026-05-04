from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    source_model = base_dir / config["training"]["model_path"]
    deployed_model = base_dir / config["deployment"]["deployed_model_path"]

    ensure_parent(deployed_model)
    shutil.copy2(source_model, deployed_model)

    print(f"Deployment completed. Model copied to {deployed_model.as_posix()}")


if __name__ == "__main__":
    main()
