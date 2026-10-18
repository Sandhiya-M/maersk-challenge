from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse
from src.common.config import MODELS_DIR
from src.ml.predict import predict_delay

router = APIRouter(prefix="/predict", tags=["predict"])


def _default_metrics() -> dict[str, object]:
    return {
        "model": "unavailable",
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "pr_auc": 0.0,
        "positive_rate": 0.0,
        "baseline_precision": 0.0,
        "baseline_recall": 0.0,
        "baseline_f1": 0.0,
        "baseline_auc": 0.0,
        "model_b_precision": 0.0,
        "model_b_recall": 0.0,
        "model_b_f1": 0.0,
        "model_b_auc": 0.0,
        "confusion_matrix": [0, 0, 0, 0],
        "calibration_table": [],
    }


@router.get("/metrics")
def get_model_metrics() -> dict[str, object]:
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics not found. Train the model first.")

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return _default_metrics()

    if isinstance(payload, dict):
        return payload
    return _default_metrics()


@router.post("", response_model=PredictionResponse)
@router.post("-delay", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    prediction = predict_delay(payload.model_dump())
    return PredictionResponse(
        shipment_id=prediction["shipment_id"],
        is_more_than_24h_late=prediction["is_more_than_24h_late"],
        probability_gt_24h_late=prediction["probability_gt_24h_late"],
        predicted_delay_hours=prediction["predicted_delay_hours"],
    )
