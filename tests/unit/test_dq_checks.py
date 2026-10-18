import json

import pandas as pd

from src.dq.checks import check_row_count, detect_dataset_issues, has_required_columns, no_duplicates, no_nulls


def test_has_required_columns() -> None:
    df = pd.DataFrame({"shipment_id": ["A"], "booking_time": ["2024-01-01"]})
    assert has_required_columns(df, ["shipment_id", "booking_time"]) is True


def test_no_nulls_and_duplicates() -> None:
    df = pd.DataFrame({"shipment_id": ["A", "B"], "booking_time": ["2024-01-01", "2024-01-02"]})
    assert no_nulls(df) is True
    assert no_duplicates(df, ["shipment_id"]) is True


def test_row_count_check() -> None:
    df = pd.DataFrame({"shipment_id": ["A", "B"]})
    assert check_row_count(df, minimum=2) is True


def test_dq_check_returns_native_bool_values() -> None:
    df = pd.DataFrame({"shipment_id": ["A", "B"], "booking_time": ["2024-01-01", "2024-01-02"]})
    result = {
        "no_duplicates": {"passed": no_duplicates(df, ["shipment_id"])},
    }
    assert isinstance(result["no_duplicates"]["passed"], bool)
    json.dumps(result)


def test_detect_dataset_issues_reports_actionable_problem() -> None:
    df = pd.DataFrame(
        {
            "shipment_id": ["SHP-1", "SHP-1", "SHP-2"],
            "port_of_loading": ["SGSIN", "SGSIN", "ZZZZZ"],
            "port_of_discharge": ["USLAX", "USLAX", "USLAX"],
            "weight_tons": [100.0, 100.0, -5.0],
            "cargo_type": ["raw materials", None, "electronics"],
        }
    )
    issues = detect_dataset_issues(df, "shipments", ["shipment_id", "port_of_loading", "port_of_discharge"], ["SGSIN", "USLAX"])
    assert any(issue["code"] == "duplicate_shipment_id" for issue in issues)
    assert any(issue["code"] == "orphan_port_code" for issue in issues)
    assert any(issue["code"] == "negative_weight_tons" for issue in issues)
