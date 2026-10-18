import pandas as pd

from src.ml.features import build_booking_features


def test_build_booking_features_contains_required_columns() -> None:
    df = pd.DataFrame(
        [{
            "shipment_id": "SHP-001",
            "booking_time": pd.Timestamp("2024-01-15 12:00:00"),
        }]
    )
    features = build_booking_features(df)
    assert set(["shipment_id", "booking_month", "booking_day_of_week", "booking_hour"]).issubset(features.columns)
