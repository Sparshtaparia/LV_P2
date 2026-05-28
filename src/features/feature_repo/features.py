from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

customer = Entity(
    name="customerID",
    join_keys=["customerID"],
    value_type=String,
)

customer_stats_source = FileSource(
    path="data/processed/model_input.parquet",
    timestamp_field="event_timestamp",
)

customer_features_view = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    source=customer_stats_source,
    schema=[
        Field(name="tenure", dtype=Float32),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
        Field(name="RFM_Score", dtype=Float32),
        Field(name="churn_probability", dtype=Float32),
    ]
)
