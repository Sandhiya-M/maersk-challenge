from __future__ import annotations

from contextlib import closing
from typing import Any

import pandas as pd
from fastapi import APIRouter

from src.common.config import CURATED_DIR, RAW_DIR
from src.common.db import DB_PATH, get_connection

router = APIRouter(prefix="/routes", tags=["routes"])


def _normalize_route_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().isoformat()
        except Exception:
            pass
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    return value


def _load_ports() -> pd.DataFrame:
    if DB_PATH.exists():
        try:
            with closing(get_connection()) as conn:
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = {str(row[0]).lower() for row in tables}
                if "ports" in table_names:
                    return conn.execute("SELECT * FROM ports ORDER BY port_code").fetch_df()
        except Exception:
            pass

    csv_path = CURATED_DIR / "ports.csv"
    if not csv_path.exists():
        csv_path = RAW_DIR / "ports.csv"
    return pd.read_csv(csv_path)


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


def _route_delay_hours(frame: pd.DataFrame) -> pd.Series:
    if "delay_hours" in frame.columns:
        return pd.to_numeric(frame["delay_hours"], errors="coerce").fillna(0.0)

    delay_hours = pd.Series(0.0, index=frame.index)

    if {"actual_arrival", "planned_arrival"}.issubset(frame.columns):
        actual_arrival = pd.to_datetime(frame["actual_arrival"], errors="coerce")
        planned_arrival = pd.to_datetime(frame["planned_arrival"], errors="coerce")
        delay_hours = (actual_arrival - planned_arrival).dt.total_seconds().div(3600).fillna(0.0)
    elif {"actual_departure", "planned_departure"}.issubset(frame.columns):
        actual_departure = pd.to_datetime(frame["actual_departure"], errors="coerce")
        planned_departure = pd.to_datetime(frame["planned_departure"], errors="coerce")
        delay_hours = (actual_departure - planned_departure).dt.total_seconds().div(3600).fillna(0.0)

    return delay_hours


@router.get("/stats")
def route_stats() -> dict[str, int | str]:
    ports = _load_ports()
    return {
        "port_count": int(len(ports)),
        "active_routes": int(ports["port_code"].nunique() if "port_code" in ports.columns else len(ports)),
        "region": str(ports["region"].mode().iloc[0]) if "region" in ports.columns and not ports["region"].empty else "GLOBAL",
    }


@router.get("/{origin}/{destination}/stats")
def route_stats_by_origin_destination(origin: str, destination: str) -> dict[str, Any]:
    shipments = _load_shipments()
    shipment_columns = {str(column).strip().lower(): column for column in shipments.columns}

    origin_col = shipment_columns.get("origin_port") or shipment_columns.get("port_of_loading")
    destination_col = shipment_columns.get("destination_port") or shipment_columns.get("port_of_discharge")
    status_col = shipment_columns.get("status") or shipment_columns.get("shipment_status")

    if origin_col is None or destination_col is None:
        return {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "average_delay_hours": 0.0,
            "on_time_rate": 0.0,
            "on_time_rate_percent": 0.0,
            "shipment_count": 0,
            "scheduled_shipment_count": 0,
            "scheduled_shipments": [],
            "shipments": [],
        }

    filtered = shipments.copy()
    filtered[origin_col] = filtered[origin_col].astype(str).str.strip().str.upper()
    filtered[destination_col] = filtered[destination_col].astype(str).str.strip().str.upper()

    route_shipments = filtered[
        (filtered[origin_col] == origin.upper()) & (filtered[destination_col] == destination.upper())
    ].copy()

    if route_shipments.empty:
        return {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "average_delay_hours": 0.0,
            "on_time_rate": 0.0,
            "on_time_rate_percent": 0.0,
            "shipment_count": 0,
            "scheduled_shipment_count": 0,
            "scheduled_shipments": [],
            "shipments": [],
        }

    delay_hours = _route_delay_hours(route_shipments)
    on_time_mask = delay_hours.le(0)

    if status_col is not None:
        status_values = route_shipments[status_col].astype(str).str.upper().str.strip()
        on_time_mask = on_time_mask | status_values.isin({"ON_TIME", "ON TIME", "ONTIME"})
        on_time_mask = on_time_mask & ~status_values.isin({"DELAYED"})

    shipment_count = int(len(route_shipments))
    average_delay_hours = float(delay_hours.mean()) if shipment_count else 0.0
    on_time_rate = float(on_time_mask.mean()) if shipment_count else 0.0

    shipment_columns = [
        "shipment_id",
        "status",
        "origin_port",
        "destination_port",
        "port_of_loading",
        "port_of_discharge",
        "booking_date",
        "planned_departure",
        "planned_arrival",
        "actual_arrival",
        "cargo_type",
    ]
    available_columns = [column for column in shipment_columns if column in route_shipments.columns]
    visible_shipments = route_shipments[available_columns].copy() if available_columns else route_shipments.copy()
    visible_shipments = visible_shipments.astype(object).where(pd.notna(visible_shipments), None)
    delay_values = _route_delay_hours(route_shipments)
    shipment_rows = visible_shipments.to_dict(orient="records")
    scheduled_shipments = []
    for idx, row in enumerate(shipment_rows):
        shipment_record = {str(key): _normalize_route_value(value) for key, value in row.items()}
        shipment_delay = float(delay_values.iloc[idx]) if idx < len(delay_values) else 0.0
        shipment_record["delay_hours"] = round(max(shipment_delay, 0.0), 2)
        scheduled_shipments.append(shipment_record)

    return {
        "origin": origin.upper(),
        "destination": destination.upper(),
        "average_delay_hours": round(average_delay_hours, 2),
        "on_time_rate": round(on_time_rate, 4),
        "on_time_rate_percent": round(on_time_rate * 100, 2),
        "shipment_count": shipment_count,
        "scheduled_shipment_count": shipment_count,
        "scheduled_shipments": scheduled_shipments,
        "shipments": scheduled_shipments,
    }
