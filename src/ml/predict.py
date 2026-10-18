from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.common.config import CURATED_DIR, MODELS_DIR
from src.ml.features import build_features
from src.ml.train import train_model


@lru_cache(maxsize=1)
def load_model(path: str | Path | None = None):
    artifact_path = Path(path) if path is not None else MODELS_DIR / "delay_model.joblib"
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        try:
            train_model(out_dir=artifact_path.parent)
        except ValueError as exc:
            raise ValueError(
                "A trained delay model could not be created from the repository's real shipment data. "
                "The dataset is too sparse or lacks both classes required for a valid booking-time model."
            ) from exc

    artifact = joblib.load(artifact_path)
    if isinstance(artifact, dict) and "model" in artifact:
        expected_columns = artifact.get("feature_columns")
        expected = {"origin_port", "destination_port", "cargo_type", "route_delay_rate", "weight_tons", "container_count", "transit_days_planned", "booking_lead_days", "booking_month", "booking_dow", "booking_hour", "origin_congestion", "destination_congestion"}
        if expected_columns is not None and set(expected_columns) != expected:
            train_model(out_dir=artifact_path.parent)
            artifact = joblib.load(artifact_path)
        return artifact

    if hasattr(artifact, "predict_proba"):
        train_model(out_dir=artifact_path.parent)
        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact
        return {"model": artifact, "threshold": 0.5, "version": "legacy"}
    raise TypeError("Model artifact is not a valid sklearn pipeline or model bundle.")


def _make_feature_frame(payload: dict[str, object]) -> pd.DataFrame:
    record = {key: value for key, value in payload.items() if value is not None}
    if "booking_date" not in record and "booking_time" in record:
        record["booking_date"] = record["booking_time"]
    if "origin_port" not in record and "port_of_loading" in record:
        record["origin_port"] = record["port_of_loading"]
    if "destination_port" not in record and "port_of_discharge" in record:
        record["destination_port"] = record["port_of_discharge"]
    if "cargo_type" not in record:
        record["cargo_type"] = "UNKNOWN"
    if "weight_tons" not in record:
        record["weight_tons"] = 0.0
    if "container_count" not in record:
        record["container_count"] = 0.0
    record.setdefault("origin_port", "UNKNOWN")
    record.setdefault("destination_port", "UNKNOWN")
    booking_timestamp = pd.to_datetime(record.get("booking_date"), errors="coerce")
    if pd.isna(booking_timestamp):
        booking_timestamp = pd.Timestamp("2024-01-01T00:00:00")
    record.setdefault("booking_date", booking_timestamp.isoformat())
    record.setdefault("planned_departure", (booking_timestamp + pd.Timedelta(days=7)).isoformat())
    record.setdefault("planned_arrival", (booking_timestamp + pd.Timedelta(days=14)).isoformat())
    record.setdefault("weight_tons", 0.0)
    record.setdefault("container_count", 0.0)
    record.setdefault("cargo_type", "UNKNOWN")

    history_path = CURATED_DIR / "shipments.csv"
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    feature_frame = build_features(pd.DataFrame([record]), history_df=history_df)
    feature_frame["route"] = (
        feature_frame["origin_port"].astype(str).str.strip().str.upper()
        + "->"
        + feature_frame["destination_port"].astype(str).str.strip().str.upper()
    )
    return feature_frame


def predict(features: dict[str, object]) -> dict[str, object]:
    bundle = load_model()
    model = bundle["model"]
    feature_frame = _make_feature_frame(features)
    probability = float(model.predict_proba(feature_frame)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    is_late = bool(probability >= threshold)
    return {
        "shipment_id": str(features.get("shipment_id", "UNKNOWN")),
        "is_more_than_24h_late": is_late,
        "probability_gt_24h_late": probability,
        "predicted_delay_hours": float(probability * 72.0),
    }


def predict_delay(payload: dict[str, object]) -> dict[str, object]:
    return predict(payload)
