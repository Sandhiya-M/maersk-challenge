from __future__ import annotations

import pandas as pd


def _with_quarantine_reason(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    result = frame.copy()
    if "quarantine_reason" not in result.columns:
        result["quarantine_reason"] = reason
    else:
        result["quarantine_reason"] = result["quarantine_reason"].fillna(reason)
    return result


def _canonical_cargo_type(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    normalized = str(value).strip().lower().replace("_", " ")
    mapping = {
        "raw materials": "Raw Materials",
        "food & beverage": "Food & Beverage",
        "food and beverage": "Food & Beverage",
        "consumer goods": "Consumer Goods",
        "automotive parts": "Automotive Parts",
        "machinery": "Machinery",
        "apparel": "Apparel",
        "electronics": "Electronics",
        "chemicals": "Chemicals",
    }
    return mapping.get(normalized, str(value).strip().title())


def _canonical_status(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    normalized = str(value).strip().lower()
    mapping = {
        "completed": "DELIVERED",
        "complete": "DELIVERED",
        "delivered": "DELIVERED",
        "in_transit": "IN_TRANSIT",
        "in transit": "IN_TRANSIT",
        "pending": "PENDING",
    }
    return mapping.get(normalized, str(value).strip().upper())


def clean_shipments(
    df: pd.DataFrame,
    valid_ports: list[str] | None = None,
    return_quarantine: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    cleaned = cleaned.rename(
        columns={
            "origin_port": "port_of_loading",
            "destination_port": "port_of_discharge",
            "booking_date": "booking_time",
        }
    )
    cleaned["shipment_id"] = cleaned["shipment_id"].astype(str).str.strip()
    cleaned["port_of_loading"] = cleaned["port_of_loading"].astype(str).str.strip().str.upper()
    cleaned["port_of_discharge"] = cleaned["port_of_discharge"].astype(str).str.strip().str.upper()
    cleaned["booking_time"] = pd.to_datetime(cleaned.get("booking_time"), errors="coerce")
    cleaned["actual_departure"] = pd.to_datetime(cleaned.get("actual_departure"), errors="coerce")
    cleaned["planned_departure"] = pd.to_datetime(cleaned.get("planned_departure"), errors="coerce")

    cleaned = cleaned.drop_duplicates(subset=["shipment_id"], keep="first")
    quarantine_frames: list[pd.DataFrame] = []

    if valid_ports is not None:
        valid_port_set = {str(port).strip().upper() for port in valid_ports}
        invalid_mask = (~cleaned["port_of_loading"].isin(valid_port_set)) | (~cleaned["port_of_discharge"].isin(valid_port_set))
        if invalid_mask.any():
            quarantine_frames.append(_with_quarantine_reason(cleaned.loc[invalid_mask].copy(), "orphan_port_code"))
        cleaned = cleaned.loc[~invalid_mask].copy()

    if "weight_tons" in cleaned.columns:
        cleaned["weight_tons"] = pd.to_numeric(cleaned["weight_tons"], errors="coerce")
        invalid_weight = cleaned["weight_tons"].le(0)
        if invalid_weight.any():
            quarantine_frames.append(_with_quarantine_reason(cleaned.loc[invalid_weight].copy(), "negative_weight_tons"))
        cleaned.loc[invalid_weight, "weight_tons"] = pd.NA

    if "container_count" in cleaned.columns:
        cleaned["container_count"] = pd.to_numeric(cleaned["container_count"], errors="coerce")
        zero_container = cleaned["container_count"].eq(0)
        if zero_container.any():
            quarantine_frames.append(_with_quarantine_reason(cleaned.loc[zero_container].copy(), "container_count_zero"))
        cleaned.loc[zero_container, "container_count"] = pd.NA

    if {"booking_time", "actual_departure"}.issubset(cleaned.columns):
        impossible_timeline = cleaned["actual_departure"] < cleaned["booking_time"]
        if impossible_timeline.any():
            quarantine_frames.append(_with_quarantine_reason(cleaned.loc[impossible_timeline].copy(), "actual_departure_before_booking_time"))
        cleaned = cleaned.loc[~impossible_timeline].copy()

    if "cargo_type" in cleaned.columns:
        cleaned["cargo_type"] = cleaned["cargo_type"].apply(_canonical_cargo_type)

    if "status" in cleaned.columns:
        cleaned["status"] = cleaned["status"].apply(_canonical_status)

    cleaned = cleaned.dropna(subset=["shipment_id"]).reset_index(drop=True)
    quarantine = pd.concat(quarantine_frames, ignore_index=True) if quarantine_frames else pd.DataFrame(columns=[*cleaned.columns, "quarantine_reason"])
    quarantine = quarantine.dropna(subset=["shipment_id"], how="all").reset_index(drop=True)

    if return_quarantine or valid_ports is not None:
        return cleaned, quarantine
    return cleaned


def clean_ports(df: pd.DataFrame, return_quarantine: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    cleaned["port_code"] = cleaned["port_code"].astype(str).str.strip().str.upper()
    cleaned["port_name"] = cleaned["port_name"].astype(str).str.strip()
    if "country" in cleaned.columns:
        cleaned.loc[cleaned["port_code"].eq("BEANR"), "country"] = "Belgium"
    invalid_port_mask = cleaned["port_code"].isna() | cleaned["port_code"].eq("")
    quarantine = cleaned.loc[invalid_port_mask].copy()
    if not quarantine.empty:
        quarantine = _with_quarantine_reason(quarantine, "invalid_port_code")
    cleaned = cleaned.loc[~invalid_port_mask].copy()
    if return_quarantine:
        return cleaned.reset_index(drop=True), quarantine.reset_index(drop=True)
    return cleaned.reset_index(drop=True)


def clean_port_events(df: pd.DataFrame, return_quarantine: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()
    cleaned = cleaned.rename(columns={"event_timestamp": "event_time"})
    cleaned["event_id"] = cleaned["event_id"].astype(str).str.strip()
    cleaned["event_time"] = pd.to_datetime(cleaned["event_time"], errors="coerce")
    cleaned["event_type"] = cleaned["event_type"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    if "vessel_id" in cleaned.columns:
        placeholder_mask = cleaned["vessel_id"].astype(str).str.upper().isin({"VSL-0000", "VSL-9999"})
        quarantine = cleaned.loc[placeholder_mask].copy()
        if not quarantine.empty:
            quarantine = _with_quarantine_reason(quarantine, "orphan_vessel_id")
        cleaned = cleaned.loc[~placeholder_mask].copy()
    else:
        quarantine = pd.DataFrame(columns=cleaned.columns)
    cleaned = cleaned.dropna(subset=["event_id", "event_time"]).reset_index(drop=True)
    quarantine = quarantine.reset_index(drop=True)
    if return_quarantine:
        return cleaned, quarantine
    return cleaned
