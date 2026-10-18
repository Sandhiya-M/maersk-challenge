from __future__ import annotations

from contextlib import closing
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query


def _to_json_safe(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.item()
            return _to_json_safe(converted)
        except (TypeError, ValueError):
            pass
    if pd.isna(value):
        return None
    return value

from src.common.config import CURATED_DIR, RAW_DIR
from src.common.db import DB_PATH, get_connection

router = APIRouter(prefix="/shipments", tags=["shipments"])


def _load_shipments() -> pd.DataFrame:
    if DB_PATH.exists():
        try:
            with closing(get_connection()) as conn:
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = {str(row[0]).lower() for row in tables}
                if "shipments" in table_names:
                    frame = conn.execute("SELECT * FROM shipments ORDER BY shipment_id").fetch_df()
                    if not frame.empty:
                        return frame
        except Exception:
            pass

    csv_path = CURATED_DIR / "shipments.csv"
    if not csv_path.exists():
        csv_path = RAW_DIR / "shipments.csv"

    if not csv_path.exists():
        return pd.DataFrame(columns=[
            "shipment_id",
            "customer_id",
            "origin_port",
            "destination_port",
            "vessel_id",
            "booking_date",
            "planned_departure",
            "actual_departure",
            "planned_arrival",
            "actual_arrival",
            "container_count",
            "cargo_type",
            "weight_tons",
            "status",
        ])

    return pd.read_csv(csv_path)


@router.get("")
def list_shipments(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=5000),
) -> list[dict[str, Any]]:
    shipments = _load_shipments()
    end = offset + limit
    records = shipments.iloc[offset:end].to_dict(orient="records")
    return [{key: _to_json_safe(value) for key, value in row.items()} for row in records]


@router.get("/{shipment_id}")
def get_shipment(shipment_id: str) -> dict[str, Any]:
    shipments = _load_shipments()
    match = shipments[shipments["shipment_id"].astype(str).str.upper() == shipment_id.upper()]
    if match.empty:
        raise HTTPException(status_code=404, detail={"code": "shipment_not_found", "message": "Shipment was not found."})
    return match.iloc[0].fillna("").to_dict()
