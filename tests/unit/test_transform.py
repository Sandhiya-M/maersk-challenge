import pandas as pd

from src.pipeline.transform import clean_shipments


def test_clean_shipments_standardizes_dates() -> None:
    df = pd.DataFrame(
        [{
            "shipment_id": "SHP-001",
            "booking_time": "2024-01-15T12:00:00",
            "port_of_loading": " SGSIN ",
            "port_of_discharge": "USLAX",
            "delay_hours": 2.5,
        }]
    )
    cleaned = clean_shipments(df)
    assert cleaned["booking_time"].dtype.kind == "M"
    assert cleaned["port_of_loading"].iloc[0] == "SGSIN"
