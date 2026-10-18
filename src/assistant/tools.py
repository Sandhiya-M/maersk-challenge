from __future__ import annotations

from typing import Any

import pandas as pd

from src.api.routers.routes import route_stats_by_origin_destination
from src.common.config import CURATED_DIR, RAW_DIR
from src.ml.predict import predict_delay as predict_shipment_delay

# Flat arguments are far easier for a small model than a nested "filters" object.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_shipments",
            "description": (
                "List shipments. Use for questions about specific shipments, statuses or counts. "
                "Ports are 5-letter codes such as CNSHA or NLRTM."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin port code, e.g. CNSHA"},
                    "destination": {"type": "string", "description": "Destination port code, e.g. NLRTM"},
                    "status": {"type": "string", "enum": ["DELIVERED", "DELAYED", "CANCELLED"]},
                    "cargo_type": {"type": "string", "description": "e.g. Electronics"},
                    "date_from": {"type": "string", "description": "Booking date from, YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "Booking date to, YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "Max rows, 1-50"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route_stats",
            "description": (
                "Average delay hours, on-time rate and shipment count for ONE route. "
                "Use for 'on-time rate from X to Y'. Arguments must be port codes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin port code, e.g. CNSHA"},
                    "destination": {"type": "string", "description": "Destination port code, e.g. NLRTM"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_delay_routes",
            "description": (
                "Routes ranked by highest average delay over the most recent quarter (last 90 days "
                "of the dataset). Use for 'which routes have the highest delay'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many routes, default 5"},
                    "min_shipments": {"type": "integer", "description": "Ignore routes with fewer shipments, default 5"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_most_delayed_shipments",
            "description": (
                "Individual SHIPMENTS ranked by largest arrival delay in hours. Use for 'which shipment "
                "was delayed the most', 'worst delays', 'longest delay'. Optional route and date filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many shipments, default 5"},
                    "origin": {"type": "string", "description": "Optional origin port code"},
                    "destination": {"type": "string", "description": "Optional destination port code"},
                    "date_from": {"type": "string", "description": "Booking date from, YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "Booking date to, YYYY-MM-DD"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_delay",
            "description": "Delay risk (probability of arriving >24h late) for one shipment ID like SHP-00421.",
            "parameters": {
                "type": "object",
                "properties": {"shipment_id": {"type": "string", "description": "e.g. SHP-00421"}},
                "required": ["shipment_id"],
            },
        },
    },
]

_COLUMN_ALIASES = {
    "port_of_loading": "origin_port",
    "port_of_discharge": "destination_port",
    "booking_time": "booking_date",
}
_DATE_COLUMNS = ["booking_date", "planned_departure", "actual_departure", "planned_arrival", "actual_arrival"]


def _read_shipments_df() -> pd.DataFrame:
    """Load shipments with canonical column names, parsed dates and duplicate IDs removed."""
    df = pd.DataFrame()
    for path in (CURATED_DIR / "shipments.csv", RAW_DIR / "shipments.csv"):
        if path.exists():
            df = pd.read_csv(path)
            break
    if df.empty:
        return df
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    for column in _DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    if "shipment_id" in df.columns:
        df = df.drop_duplicates(subset="shipment_id", keep="first")
    return df


class DataUnavailableError(RuntimeError):
    """Raised when shipment data is missing or lacks the columns a tool needs."""


def _require_columns(df: pd.DataFrame, needed: set[str]) -> None:
    if df.empty:
        raise DataUnavailableError(
            f"No shipment data was loaded. Looked for {CURATED_DIR / 'shipments.csv'} and {RAW_DIR / 'shipments.csv'}."
        )
    missing = sorted(needed - set(df.columns))
    if missing:
        raise DataUnavailableError(
            f"Shipment data is missing columns {missing}. Columns found: {list(df.columns)}"
        )


def data_window() -> dict[str, str | None]:
    """Min and max booking date in the data, used in the system prompt."""
    df = _read_shipments_df()
    if df.empty or "booking_date" not in df.columns or df["booking_date"].dropna().empty:
        return {"min": None, "max": None}
    return {
        "min": df["booking_date"].min().strftime("%Y-%m-%d"),
        "max": df["booking_date"].max().strftime("%Y-%m-%d"),
    }


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def query_shipments(filters: dict[str, Any] | None = None, limit: int = 25) -> list[dict[str, Any]]:
    df = _read_shipments_df()
    if df.empty:
        return []
    working = df.copy()
    f = {str(k).strip().lower(): v for k, v in (filters or {}).items() if v not in (None, "")}
    key_map = {"origin": "origin_port", "destination": "destination_port"}
    for key, value in f.items():
        column = key_map.get(key, key)
        if key in {"date_from", "date_to"} or column not in working.columns:
            continue
        working = working[working[column].astype(str).str.strip().str.upper() == str(value).strip().upper()]
    if "date_from" in f:
        working = working[working["booking_date"] >= pd.to_datetime(f["date_from"], errors="coerce")]
    if "date_to" in f:
        working = working[working["booking_date"] <= pd.to_datetime(f["date_to"], errors="coerce")]
    limit = max(1, min(int(limit), 50))
    rows = working.head(limit).to_dict(orient="records")
    return [{str(c): _normalize_value(v) for c, v in row.items()} for row in rows]


def get_route_stats(origin: str, destination: str) -> dict[str, Any]:
    result = route_stats_by_origin_destination(str(origin).strip().upper(), str(destination).strip().upper())
    if result.get("shipment_count", 0) == 0:
        return {
            "origin": str(origin).upper(),
            "destination": str(destination).upper(),
            "average_delay_hours": 0.0,
            "on_time_rate": 0.0,
            "on_time_rate_percent": 0.0,
            "shipment_count": 0,
            "scheduled_shipments": [],
        }
    return result


def get_top_delay_routes(period: str = "quarter", limit: int = 5, min_shipments: int = 5) -> list[dict[str, Any]]:
    """Top routes by average arrival delay in the last 90 days of the dataset."""
    df = _read_shipments_df()
    needed = {"origin_port", "destination_port", "planned_arrival", "actual_arrival", "booking_date"}
    _require_columns(df, needed)
    cutoff = df["booking_date"].max() - pd.Timedelta(days=90)
    working = df[df["booking_date"] >= cutoff].copy()
    working["delay_hours"] = (working["actual_arrival"] - working["planned_arrival"]).dt.total_seconds() / 3600
    working = working.dropna(subset=["origin_port", "destination_port", "delay_hours"])
    if working.empty:
        return []
    working["route"] = working["origin_port"].astype(str).str.upper() + "->" + working["destination_port"].astype(str).str.upper()
    grouped = (
        working.groupby("route")["delay_hours"].agg(average_delay_hours="mean", shipment_count="count").reset_index()
    )
    filtered = grouped[grouped["shipment_count"] >= int(min_shipments)]
    grouped = filtered if not filtered.empty else grouped  # never return empty just because of the minimum
    grouped = grouped.sort_values(["average_delay_hours", "shipment_count"], ascending=[False, False]).head(
        max(1, min(int(limit), 20))
    )
    out = []
    for _, row in grouped.iterrows():
        origin, destination = str(row["route"]).split("->", 1)
        out.append({
            "route": row["route"],
            "origin": origin,
            "destination": destination,
            "average_delay_hours": round(float(row["average_delay_hours"]), 2),
            "shipment_count": int(row["shipment_count"]),
        })
    return out


def get_most_delayed_shipments(
    limit: int = 5,
    origin: str | None = None,
    destination: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Shipments with the largest (actual_arrival - planned_arrival) in hours."""
    df = _read_shipments_df()
    needed = {"shipment_id", "origin_port", "destination_port", "planned_arrival", "actual_arrival"}
    _require_columns(df, needed)
    working = df.copy()
    if origin:
        working = working[working["origin_port"].astype(str).str.upper() == str(origin).strip().upper()]
    if destination:
        working = working[working["destination_port"].astype(str).str.upper() == str(destination).strip().upper()]
    if date_from:
        working = working[working["booking_date"] >= pd.to_datetime(date_from, errors="coerce")]
    if date_to:
        working = working[working["booking_date"] <= pd.to_datetime(date_to, errors="coerce")]
    working = working.copy()
    working["delay_hours"] = (working["actual_arrival"] - working["planned_arrival"]).dt.total_seconds() / 3600
    working = working.dropna(subset=["delay_hours"]).sort_values("delay_hours", ascending=False)
    out = []
    for _, row in working.head(max(1, min(int(limit), 20))).iterrows():
        out.append({
            "shipment_id": row["shipment_id"],
            "route": f"{row['origin_port']}->{row['destination_port']}",
            "delay_hours": round(float(row["delay_hours"]), 1),
            "planned_arrival": _normalize_value(row["planned_arrival"]),
            "actual_arrival": _normalize_value(row["actual_arrival"]),
            "status": _normalize_value(row.get("status")),
        })
    return out


def predict_delay(shipment_id: str) -> dict[str, Any]:
    df = _read_shipments_df()
    match = df[df["shipment_id"].astype(str).str.upper() == str(shipment_id).strip().upper()]
    if match.empty:
        raise ValueError(f"Shipment {shipment_id!r} not found in the curated data product.")
    row = match.iloc[0].to_dict()
    payload = {
        "shipment_id": row.get("shipment_id"),
        "origin_port": row.get("origin_port"),
        "destination_port": row.get("destination_port"),
        "cargo_type": row.get("cargo_type"),
        "weight_tons": row.get("weight_tons"),
        "container_count": row.get("container_count"),
        "booking_date": row.get("booking_date"),
        "planned_departure": row.get("planned_departure"),
        "planned_arrival": row.get("planned_arrival"),
    }
    return predict_shipment_delay(payload)


def execute_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool. Raises on unknown tools or bad input (kept for existing tests)."""
    args = dict(arguments or {})
    if name == "query_shipments":
        filters = dict(args.pop("filters", None) or {})
        for key in ("origin", "destination", "status", "cargo_type", "date_from", "date_to"):
            if args.get(key):
                filters[key] = args[key]
        return {"tool": name, "result": query_shipments(filters, limit=int(args.get("limit", 25)))}
    if name == "get_route_stats":
        return {"tool": name, "result": get_route_stats(args.get("origin", ""), args.get("destination", ""))}
    if name == "get_top_delay_routes":
        return {
            "tool": name,
            "result": get_top_delay_routes(
                period=str(args.get("period", "quarter")),
                limit=int(args.get("limit", 5)),
                min_shipments=int(args.get("min_shipments", 5)),
            ),
        }
    if name == "get_most_delayed_shipments":
        return {
            "tool": name,
            "result": get_most_delayed_shipments(
                limit=int(args.get("limit", 5)),
                origin=args.get("origin"),
                destination=args.get("destination"),
                date_from=args.get("date_from"),
                date_to=args.get("date_to"),
            ),
        }
    if name == "predict_delay":
        return {"tool": name, "result": predict_delay(args.get("shipment_id", ""))}
    raise ValueError(f"Unsupported tool: {name}")


def safe_execute_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Like execute_tool_call, but returns errors as data so the model can react to them."""
    try:
        out = execute_tool_call(name, arguments)
    except Exception as exc:  # noqa: BLE001 - tool errors must go back to the model
        return {"tool": name, "result": {"error": {"code": type(exc).__name__, "message": str(exc)}}}
    result = out["result"]
    if name == "get_route_stats" and result.get("shipment_count", 0) == 0:
        return {"tool": name, "result": {"error": {
            "code": "NO_SHIPMENTS_ON_ROUTE",
            "message": "No shipments found. Check that origin and destination are valid 5-letter port codes.",
        }}}
    if name in {"get_top_delay_routes", "get_most_delayed_shipments"} and not result:
        return {"tool": name, "result": {"error": {
            "code": "NO_DATA",
            "message": "The query returned no rows (filters matched nothing, or no shipment has both planned and actual arrival dates). This is a data or filter problem, not proof that there were no delays.",
        }}}
    if name == "query_shipments":
        compact = [
            {k: r.get(k) for k in ("shipment_id", "origin_port", "destination_port", "cargo_type",
                                   "status", "booking_date", "planned_arrival", "actual_arrival")}
            for r in result
        ]
        out = {"tool": name, "result": {"rows_returned": len(compact), "shipments": compact}}
    return out


__all__ = [
    "query_shipments",
    "get_route_stats",
    "get_top_delay_routes",
    "get_most_delayed_shipments",
    "predict_delay",
    "execute_tool_call",
    "safe_execute_tool_call",
    "data_window",
    "TOOL_DEFINITIONS",
]