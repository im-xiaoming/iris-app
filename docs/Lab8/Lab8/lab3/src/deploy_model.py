from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "params.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def clone_model_version(source_model: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model, target_dir / source_model.name)


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]

    model_path = base_dir / config["artifacts"]["model_path"]
    registry_dir = base_dir / config["deployment"]["registry_dir"]
    current_link = base_dir / config["deployment"]["current_link"]
    previous_link = base_dir / config["deployment"]["previous_link"]

    version_name = datetime.now().strftime("model_%Y%m%d_%H%M%S")
    new_version_dir = registry_dir / version_name
    clone_model_version(model_path, new_version_dir)

    current_target = current_link.resolve(strict=False) if current_link.exists() else None
    if current_target and current_target.exists():
        safe_remove(previous_link)
        previous_link.parent.mkdir(parents=True, exist_ok=True)
        previous_link.symlink_to(current_target, target_is_directory=True)

    safe_remove(current_link)
    current_link.parent.mkdir(parents=True, exist_ok=True)
    current_link.symlink_to(new_version_dir.resolve(), target_is_directory=True)

    print(f"Deployed version: {new_version_dir.as_posix()}")
    print(f"Current deployment pointer: {current_link.as_posix()}")


if __name__ == "__main__":
    main()
