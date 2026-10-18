from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.common.config import MODELS_DIR
from src.ml.features import build_delay_target, build_features
from src.ml.train import _load_training_data, time_based_split


def evaluate_model(db_path: str | Path | None = None, out_dir: str | Path = MODELS_DIR) -> dict[str, object]:
    df = _load_training_data(db_path)
    if "is_delayed_label" not in df.columns:
        df["is_delayed_label"] = build_delay_target(df)
    df = df.drop(columns=[column for column in ["actual_departure", "actual_arrival", "actual_delay_hours", "on_time_flag", "is_delayed_label", "status", "transit_days_actual", "departure_delay_hours"] if column in df.columns], errors="ignore")

    train_df, val_df, test_df = time_based_split(df)
    model_bundle = joblib.load(Path(out_dir) / "delay_model.joblib")
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))

    X_train = build_features(train_df, history_df=train_df)
    X_val = build_features(val_df, history_df=train_df)
    X_test = build_features(test_df, history_df=train_df)

    X_train["route"] = X_train["origin_port"].astype(str) + "->" + X_train["destination_port"].astype(str)
    X_val["route"] = X_val["origin_port"].astype(str) + "->" + X_val["destination_port"].astype(str)
    X_test["route"] = X_test["origin_port"].astype(str) + "->" + X_test["destination_port"].astype(str)

    y_test = build_delay_target(test_df)
    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": [int(v) for v in confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()],
        "positive_rate": float(y_test.mean()),
        "threshold": float(threshold),
        "calibration_bins": [],
    }

    bin_edges = np.linspace(0.0, 1.0, 11)
    for idx in range(len(bin_edges) - 1):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        mask = (probabilities >= lo) & (probabilities < hi)
        if idx == len(bin_edges) - 2:
            mask = (probabilities >= lo) & (probabilities <= hi)
        if mask.any():
            metrics["calibration_bins"].append(
                {
                    "bin": f"[{lo:.2f},{hi:.2f}]",
                    "count": int(mask.sum()),
                    "avg_predicted_probability": float(probabilities[mask].mean()),
                    "observed_rate": float(y_test.iloc[mask].mean()),
                }
            )

    out_path = Path(out_dir) / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate_model(), indent=2))
