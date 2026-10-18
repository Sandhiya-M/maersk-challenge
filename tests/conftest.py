from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_shipments_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shipment_id": "SHP-001",
                "booking_time": "2024-01-15T12:00:00",
                "port_of_loading": "SGSIN",
                "port_of_discharge": "USLAX",
                "delay_hours": 2.5,
            },
            {
                "shipment_id": "SHP-002",
                "booking_time": "2024-01-16T08:00:00",
                "port_of_loading": "NLRTM",
                "port_of_discharge": "DEHAM",
                "delay_hours": 0.0,
            },
        ]
    )
