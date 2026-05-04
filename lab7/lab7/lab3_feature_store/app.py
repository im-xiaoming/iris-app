from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from feast import FeatureStore

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT / "feature_repo"

app = FastAPI(title="Credit Feature Retrieval API")
store = FeatureStore(repo_path=str(REPO_ROOT))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/features/{customer_id}")
def get_features(customer_id: int) -> dict:
    response = store.get_online_features(
        features=[
            "credit_stats:income",
            "credit_stats:loan_amount",
            "credit_stats:age",
            "credit_stats:late_payments",
            "credit_stats:credit_utilization",
            "credit_stats:employment_years",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    payload_only = [values for key, values in response.items() if key != "customer_id"]
    if not response or all(values == [None] for values in payload_only):
        raise HTTPException(status_code=404, detail="Khong tim thay feature cho customer_id nay")

    return response
