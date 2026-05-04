from __future__ import annotations

import sys
from pathlib import Path

from kfp import compiler

CURRENT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = CURRENT_DIR.parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from automated_training_pipeline import automated_training_pipeline

PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "automated_training_pipeline.yaml"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=automated_training_pipeline,
        package_path=str(OUTPUT_PATH),
    )
    print(f"Da compile pipeline ra: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
