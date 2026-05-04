from __future__ import annotations

from datetime import datetime
from pathlib import Path

from feast import FeatureStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "feature_repo"


def main() -> None:
    store = FeatureStore(repo_path=str(REPO_ROOT))
    store.materialize_incremental(end_date=datetime.utcnow())
    print("Da materialize feature vao online store.")


if __name__ == "__main__":
    main()
