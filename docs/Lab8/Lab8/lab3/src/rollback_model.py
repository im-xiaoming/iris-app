from __future__ import annotations

import shutil
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


def main() -> None:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]

    current_link = base_dir / config["deployment"]["current_link"]
    previous_link = base_dir / config["deployment"]["previous_link"]

    if not previous_link.exists():
        raise SystemExit("Rollback failed: no previous deployed version found.")

    previous_target = previous_link.resolve()
    safe_remove(current_link)
    current_link.symlink_to(previous_target, target_is_directory=True)

    print(f"Rollback completed. Current now points to: {previous_target.as_posix()}")


if __name__ == "__main__":
    main()
