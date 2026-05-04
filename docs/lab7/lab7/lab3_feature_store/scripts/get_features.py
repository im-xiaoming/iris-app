from __future__ import annotations

from pathlib import Path

from feast import FeatureStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "feature_repo"


def main() -> None:
    store = FeatureStore(repo_path=str(REPO_ROOT))
    features = store.get_online_features(
        features=[
            "credit_stats:income",
            "credit_stats:loan_amount",
            "credit_stats:late_payments",
            "credit_stats:credit_utilization",
        ],
        entity_rows=[{"customer_id": 1001}],
    ).to_dict()
    print(features)


if __name__ == "__main__":
    main()
