from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

ROOT = Path(__file__).resolve().parents[1]

credit_source = FileSource(
    path=str(ROOT / "data" / "credit_scoring.csv"),
    timestamp_field="event_timestamp",
)

customer = Entity(name="customer_id", join_keys=["customer_id"])

credit_stats_fv = FeatureView(
    name="credit_stats",
    entities=[customer],
    ttl=None,
    schema=[
        Field(name="income", dtype=Float32),
        Field(name="loan_amount", dtype=Float32),
        Field(name="age", dtype=Int64),
        Field(name="late_payments", dtype=Int64),
        Field(name="credit_utilization", dtype=Float32),
        Field(name="employment_years", dtype=Int64),
    ],
    source=credit_source,
)
