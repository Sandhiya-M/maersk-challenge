from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BANNED_FEATURES = [
    "actual_departure",
    "actual_arrival",
    "actual_delay_hours",
    "on_time_flag",
    "is_delayed_label",
    "status",
    "transit_days_actual",
    "departure_delay_hours",
    "port_events",
]

ALLOWED_FEATURES = [
    "origin_port",
    "destination_port",
    "cargo_type",
    "route_delay_rate",
    "weight_tons",
    "container_count",
    "transit_days_planned",
    "booking_lead_days",
    "booking_month",
    "booking_dow",
    "booking_hour",
    "origin_congestion",
    "destination_congestion",
]


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _safe_numeric_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def _route_delay_series(frame: pd.DataFrame, label_name: str = "is_delayed_label", alpha: float = 10.0) -> pd.Series:
    base = frame.copy()
    if label_name not in base.columns:
        if {"planned_arrival", "actual_arrival"}.issubset(base.columns):
            base[label_name] = build_delay_target(base)
        else:
            base[label_name] = 0.0
    base = base[["route", label_name]].copy()
    base[label_name] = pd.to_numeric(base[label_name], errors="coerce").fillna(0.0)
    global_rate = _safe_numeric_mean(base[label_name]) if not base.empty else 0.0
    route_summary = base.groupby("route", dropna=False)[label_name].agg(["count", "sum"]).reset_index()
    route_summary["route_delay_rate"] = (
        route_summary["sum"] + alpha * global_rate
    ) / (route_summary["count"] + alpha)
    lookup = route_summary.set_index("route")["route_delay_rate"]
    return base["route"].map(lookup).fillna(global_rate)


class RouteDelayRateEncoder(BaseEstimator, TransformerMixin):
    """Bayesian-smoothed route delay rate computed from training rows only."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.global_rate_ = 0.0
        self.route_rates_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RouteDelayRateEncoder":
        frame = pd.DataFrame(X).copy()
        if "route" not in frame.columns:
            raise ValueError("RouteDelayRateEncoder requires a 'route' column.")

        if y is not None:
            labels = pd.Series(y).reset_index(drop=True)
        elif "is_delayed_label" in frame.columns:
            labels = pd.Series(frame["is_delayed_label"]).reset_index(drop=True)
        else:
            raise ValueError("RouteDelayRateEncoder needs a target when fitting.")

        route_df = pd.concat([frame[["route"]].reset_index(drop=True), labels.rename("_target").reset_index(drop=True)], axis=1)
        route_df["_target"] = pd.to_numeric(route_df["_target"], errors="coerce").fillna(0.0)
        self.global_rate_ = _safe_numeric_mean(route_df["_target"]) if not route_df.empty else 0.0
        summary = route_df.groupby("route", dropna=False)["_target"].agg(["count", "sum"]).reset_index()
        summary["route_delay_rate"] = (summary["sum"] + self.alpha * self.global_rate_) / (summary["count"] + self.alpha)
        self.route_rates_ = summary.set_index("route")["route_delay_rate"].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        if "route" not in frame.columns:
            raise ValueError("RouteDelayRateEncoder requires a 'route' column for transformation.")
        rates = frame["route"].map(self.route_rates_).fillna(self.global_rate_)
        return pd.DataFrame({"route_delay_rate": rates.astype(float).to_numpy()})


def build_delay_target(df: pd.DataFrame) -> pd.Series:
    frame = df.copy()
    if "is_delayed_label" in frame.columns:
        values = pd.to_numeric(frame["is_delayed_label"], errors="coerce")
        return values.astype("Int64")

    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    if {"planned_arrival", "actual_arrival"}.issubset(frame.columns):
        planned = _safe_datetime(frame["planned_arrival"])
        actual = _safe_datetime(frame["actual_arrival"])
        valid_mask = planned.notna() & actual.notna()
        if valid_mask.any():
            delay_hours = (actual.loc[valid_mask] - planned.loc[valid_mask]).dt.total_seconds().div(3600)
            result.loc[valid_mask] = (delay_hours > 24).astype("Int64")

    if "status" in frame.columns:
        status_values = frame["status"].astype(str).str.strip().str.upper()
        delayed_mask = status_values.isin({"DELAYED", "LATE", "LATE_ARRIVAL", "ARRIVAL_DELAYED", "DELAY"})
        on_time_mask = status_values.isin({"DELIVERED", "ON_TIME", "ON-TIME", "ON TIME", "IN_TIME", "ARRIVED", "ON-TIME-ARRIVAL"})
        if delayed_mask.any():
            result.loc[delayed_mask & result.isna()] = 1
        if on_time_mask.any():
            result.loc[on_time_mask & result.isna()] = 0

    return result


def build_features(df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = df.copy()
    if any(column in BANNED_FEATURES for column in frame.columns):
        raise ValueError(f"The input dataframe contains banned feature columns: {sorted(set(frame.columns) & set(BANNED_FEATURES))}")

    if history_df is None:
        history_df = frame
    history = history_df.copy()

    if "origin_port" in frame.columns:
        origin_series = frame["origin_port"]
    elif "port_of_loading" in frame.columns:
        origin_series = frame["port_of_loading"]
    else:
        origin_series = pd.Series("UNKNOWN", index=frame.index)

    if "destination_port" in frame.columns:
        destination_series = frame["destination_port"]
    elif "port_of_discharge" in frame.columns:
        destination_series = frame["port_of_discharge"]
    else:
        destination_series = pd.Series("UNKNOWN", index=frame.index)

    cargo_series = frame["cargo_type"] if "cargo_type" in frame.columns else pd.Series("UNKNOWN", index=frame.index)

    booking_date = (
        frame["booking_date"]
        if "booking_date" in frame.columns
        else frame["booking_time"]
        if "booking_time" in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    booking_date = _safe_datetime(booking_date)

    planned_arrival = frame["planned_arrival"] if "planned_arrival" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    planned_departure = frame["planned_departure"] if "planned_departure" in frame.columns else pd.Series(pd.NaT, index=frame.index)
    planned_arrival_ts = _safe_datetime(planned_arrival)
    planned_departure_ts = _safe_datetime(planned_departure)

    route = (
        origin_series.astype(str).str.strip().str.upper().fillna("UNKNOWN")
        + "->"
        + destination_series.astype(str).str.strip().str.upper().fillna("UNKNOWN")
    )

    history_route = (
        history["origin_port"].astype(str).str.strip().str.upper().fillna("UNKNOWN")
        + "->"
        + history["destination_port"].astype(str).str.strip().str.upper().fillna("UNKNOWN")
    ) if {"origin_port", "destination_port"}.issubset(history.columns) else route

    if "is_delayed_label" not in history.columns:
        history = history.copy()
        history["is_delayed_label"] = build_delay_target(history)
    history["route"] = history_route

    route_rates = _route_delay_series(history, label_name="is_delayed_label")
    frame["route"] = route.astype(str)
    history_mean = _safe_numeric_mean(history["is_delayed_label"]) if not history.empty else 0.0
    frame["route_delay_rate"] = frame["route"].map(route_rates).fillna(history_mean)

    frame["origin_port"] = origin_series.astype(str).str.strip().str.upper().fillna("UNKNOWN")
    frame["destination_port"] = destination_series.astype(str).str.strip().str.upper().fillna("UNKNOWN")
    frame["cargo_type"] = cargo_series.astype(str).str.strip().fillna("UNKNOWN")

    weight_series = frame["weight_tons"] if "weight_tons" in frame.columns else pd.Series(0.0, index=frame.index)
    container_series = frame["container_count"] if "container_count" in frame.columns else pd.Series(0.0, index=frame.index)
    frame["weight_tons"] = pd.to_numeric(weight_series, errors="coerce").fillna(0.0)
    frame["container_count"] = pd.to_numeric(container_series, errors="coerce").fillna(0.0)

    transit_days = (planned_arrival_ts - planned_departure_ts).dt.total_seconds().div(86400)
    transit_days = transit_days.where(transit_days.notna(), (planned_arrival_ts - booking_date).dt.total_seconds().div(86400))
    frame["transit_days_planned"] = pd.to_numeric(transit_days, errors="coerce").fillna(0.0)

    booking_lead = (planned_departure_ts - booking_date).dt.total_seconds().div(86400)
    booking_lead = booking_lead.where(booking_lead.notna(), (planned_arrival_ts - booking_date).dt.total_seconds().div(86400))
    frame["booking_lead_days"] = pd.to_numeric(booking_lead, errors="coerce").fillna(0.0)

    frame["booking_month"] = booking_date.dt.month.fillna(0).astype(int)
    frame["booking_dow"] = booking_date.dt.dayofweek.fillna(0).astype(int)
    frame["booking_hour"] = booking_date.dt.hour.fillna(0).astype(int)

    if "origin_congestion" in frame.columns:
        frame["origin_congestion"] = pd.to_numeric(frame["origin_congestion"], errors="coerce").fillna(0.0)
    else:
        frame["origin_congestion"] = 0.0

    if "destination_congestion" in frame.columns:
        frame["destination_congestion"] = pd.to_numeric(frame["destination_congestion"], errors="coerce").fillna(0.0)
    else:
        frame["destination_congestion"] = 0.0

    feature_frame = frame[ALLOWED_FEATURES].copy()
    if any(column in BANNED_FEATURES for column in feature_frame.columns):
        raise ValueError(f"Feature builder produced banned columns: {sorted(set(feature_frame.columns) & set(BANNED_FEATURES))}")
    return feature_frame


def build_booking_features(df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compatibility wrapper for the historical booking feature builder.

    The project currently uses canonical `booking_dow` naming internally, but older callers and tests
    expect `booking_day_of_week` and the raw `shipment_id` column to remain in the returned frame.
    """
    features = build_features(df, history_df=history_df).copy()
    if "shipment_id" in df.columns:
        features.insert(0, "shipment_id", df["shipment_id"].reset_index(drop=True))
    if "booking_dow" in features.columns:
        features = features.rename(columns={"booking_dow": "booking_day_of_week"})
    return features
