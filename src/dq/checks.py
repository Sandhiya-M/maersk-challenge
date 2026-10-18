from __future__ import annotations

from typing import Any

import pandas as pd


def has_required_columns(df: Any, required: list[str]) -> bool:
    missing = [col for col in required if col not in getattr(df, "columns", [])]
    return not missing


def no_nulls(df: Any, columns: list[str] | None = None) -> bool:
    target = df.columns if columns is None else columns
    return bool(df[target].isnull().sum().sum() == 0)


def no_duplicates(df: Any, key_columns: list[str]) -> bool:
    return bool(df.duplicated(subset=key_columns).sum() == 0)


def numeric_range_ok(df: Any, column: str, low: float, high: float) -> bool:
    series = df[column]
    return bool(series.between(low, high).all())


def check_row_count(df: Any, minimum: int = 1) -> bool:
    return bool(len(df) >= minimum)


def detect_dataset_issues(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str] | None = None,
    valid_ports: list[str] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required = required_columns or []

    if required and not has_required_columns(df, required):
        missing = [col for col in required if col not in df.columns]
        issues.append(
            {
                "code": "missing_required_columns",
                "severity": "critical",
                "action": "quarantine_dataset",
                "details": {"missing": missing},
            }
        )

    if "shipment_id" in df.columns:
        duplicate_mask = df["shipment_id"].duplicated(keep=False)
        if duplicate_mask.any():
            issues.append(
                {
                    "code": "duplicate_shipment_id",
                    "severity": "critical",
                    "rows": int(duplicate_mask.sum()),
                    "action": "drop_exact_duplicates_keep_most_complete",
                }
            )

    if "weight_tons" in df.columns:
        negative_rows = df["weight_tons"].le(0)
        if negative_rows.any():
            issues.append(
                {
                    "code": "negative_weight_tons",
                    "severity": "critical",
                    "rows": int(negative_rows.sum()),
                    "action": "set_null_and_flag",
                }
            )

    if "cargo_type" in df.columns:
        missing_cargo = df["cargo_type"].isna()
        if missing_cargo.any():
            issues.append(
                {
                    "code": "missing_cargo_type",
                    "severity": "info",
                    "rows": int(missing_cargo.sum()),
                    "action": "set_unknown",
                }
            )

    if "status" in df.columns:
        missing_status = df["status"].isna()
        if missing_status.any():
            issues.append(
                {
                    "code": "missing_status",
                    "severity": "warning",
                    "rows": int(missing_status.sum()),
                    "action": "set_unknown",
                }
            )

    if dataset_name == "shipments" and valid_ports is not None:
        valid = {str(port).strip().upper() for port in valid_ports}
        invalid_origin = ~df.get("port_of_loading", pd.Series(index=df.index, dtype="object")).astype(str).str.upper().isin(valid)
        invalid_destination = ~df.get("port_of_discharge", pd.Series(index=df.index, dtype="object")).astype(str).str.upper().isin(valid)
        invalid_port_rows = invalid_origin | invalid_destination
        if invalid_port_rows.any():
            issues.append(
                {
                    "code": "orphan_port_code",
                    "severity": "critical",
                    "rows": int(invalid_port_rows.sum()),
                    "action": "quarantine_orphan_ports",
                }
            )

    if dataset_name == "port_events" and "event_type" in df.columns:
        missing_event_type = df["event_type"].isna()
        if missing_event_type.any():
            issues.append(
                {
                    "code": "missing_event_type",
                    "severity": "warning",
                    "rows": int(missing_event_type.sum()),
                    "action": "set_unknown",
                }
            )

    if dataset_name == "port_events" and "event_id" in df.columns:
        duplicate_event_ids = df["event_id"].duplicated(keep=False)
        if duplicate_event_ids.any():
            issues.append(
                {
                    "code": "duplicate_event_id",
                    "severity": "critical",
                    "rows": int(duplicate_event_ids.sum()),
                    "action": "assign_surrogate_key_and_flag",
                }
            )

    return issues


def summarize_dataset_quality(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str] | None = None,
    valid_ports: list[str] | None = None,
) -> dict[str, Any]:
    normalized_df = df.copy()
    alias_map = {
        "origin_port": "port_of_loading",
        "destination_port": "port_of_discharge",
        "booking_date": "booking_time",
    }
    normalized_df = normalized_df.rename(columns={key: value for key, value in alias_map.items() if key in normalized_df.columns})
    issues = detect_dataset_issues(normalized_df, dataset_name, required_columns=required_columns, valid_ports=valid_ports)
    checks: dict[str, Any] = {
        "required_columns": {
            "passed": has_required_columns(normalized_df, required_columns or []),
            "details": {"required": required_columns or [], "available": list(normalized_df.columns)},
        }
    }
    if "shipment_id" in normalized_df.columns:
        checks["duplicate_shipment_id"] = {
            "passed": no_duplicates(normalized_df, ["shipment_id"]),
            "details": {"duplicate_rows": int(normalized_df["shipment_id"].duplicated().sum())},
        }
    checks["null_check"] = {
        "passed": no_nulls(normalized_df),
        "details": {"null_rows": int(normalized_df.isnull().any(axis=1).sum())},
    }
    return {
        "dataset": dataset_name,
        "passed": not issues,
        "issues": issues,
        "issue_count": len(issues),
        "row_count": len(normalized_df),
        "checks": checks,
    }
