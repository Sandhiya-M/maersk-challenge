from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import CURATED_DIR, MODELS_DIR, RAW_DIR
from src.ml.features import ALLOWED_FEATURES, BANNED_FEATURES, RouteDelayRateEncoder, build_delay_target, build_features


def _load_training_data(db_path: str | Path | None = None) -> pd.DataFrame:
    if db_path is not None:
        db_file = Path(db_path)
        if db_file.exists() and db_file.suffix == ".duckdb":
            try:
                import duckdb

                with closing(duckdb.connect(str(db_file), read_only=True)) as con:
                    feature_table = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'serving' AND table_name = 'shipment_features'").fetchone()
                    if feature_table:
                        df = con.execute("SELECT * FROM serving.shipment_features").fetchdf()
                        if "is_delayed_label" not in df.columns:
                            labels = con.execute("SELECT shipment_id, is_delayed_label FROM serving.shipment_labels").fetchdf()
                            if not labels.empty:
                                df = df.merge(labels, on="shipment_id", how="left")
                        if not df.empty:
                            return df
            except Exception:
                pass

    csv_candidates = [
        Path(db_path) if db_path is not None and Path(db_path).suffix.lower() != ".duckdb" else None,
        CURATED_DIR / "shipments.csv",
        RAW_DIR / "shipments.csv",
    ]
    for candidate in csv_candidates:
        if candidate is None:
            continue
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError("No training dataset was found for ML training.")


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = df.copy()
    booking_date = frame["booking_date"] if "booking_date" in frame.columns else frame["booking_time"]
    frame["booking_date"] = pd.to_datetime(booking_date, errors="coerce")
    frame = frame.dropna(subset=["booking_date"]).sort_values("booking_date").reset_index(drop=True)
    if len(frame) < 3:
        raise ValueError("Need at least three rows to create train, validation, and test splits.")
    n_train = int(len(frame) * 0.8)
    train_full = frame.iloc[:n_train].copy()
    test = frame.iloc[n_train:].copy()
    val_size = max(1, int(len(train_full) * 0.2))
    val = train_full.iloc[-val_size:].copy()
    train = train_full.iloc[:-val_size].copy() if len(train_full) > val_size else train_full.iloc[:max(1, len(train_full) - 1)].copy()
    return train, val, test


def _make_numeric_and_categorical_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
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
    categorical = ["origin_port", "destination_port", "cargo_type"]
    return [col for col in numeric if col in frame.columns], [col for col in categorical if col in frame.columns]


def _score_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_objective = -np.inf

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        false_positive_rate = fp / max(fp + tn, 1)
        precision_penalty = max(0.0, 0.50 - precision) * 2.0
        objective = f1 - (1.5 * false_positive_rate) - precision_penalty

        if tp == 0 and fp > 0:
            objective -= 2.0
        if recall < 0.2 and false_positive_rate > 0.8:
            objective -= 1.0

        if objective > best_objective or (np.isclose(objective, best_objective) and threshold > best_threshold):
            best_objective = objective
            best_threshold = float(threshold)

    if best_threshold < 0.5 and best_objective <= 0.0:
        best_threshold = 0.5

    return best_threshold, best_objective


def _validate_binary_targets(name: str, y: pd.Series) -> pd.Series:
    values = pd.Series(y).dropna().astype(int)
    if values.empty:
        raise ValueError(f"{name} target is empty after filtering missing labels.")
    if values.nunique() < 2:
        raise ValueError(
            f"{name} target has only one class after filtering ({values.unique().tolist()}). "
            "A valid binary delay model requires both delayed and on-time shipments in each split."
        )
    return values


def _build_model_pipeline() -> Pipeline:
    route_transformer = Pipeline([
        ("route_encoder", RouteDelayRateEncoder(alpha=10.0)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("route_rate", route_transformer, ["route"]),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                [
                    "weight_tons",
                    "container_count",
                    "transit_days_planned",
                    "booking_lead_days",
                    "booking_month",
                    "booking_dow",
                    "booking_hour",
                    "origin_congestion",
                    "destination_congestion",
                ],
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                ["origin_port", "destination_port", "cargo_type"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])


def _build_model_b() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                [
                    "weight_tons",
                    "container_count",
                    "transit_days_planned",
                    "booking_lead_days",
                    "booking_month",
                    "booking_dow",
                    "booking_hour",
                    "origin_congestion",
                    "destination_congestion",
                ],
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                ["origin_port", "destination_port", "cargo_type"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, class_weight="balanced")),
    ])


def _prepare_model_frame(frame: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    features = build_features(frame, history_df=history_df)
    features = features.copy()
    if {"origin_port", "destination_port"}.issubset(features.columns):
        features["route"] = features["origin_port"].astype(str) + "->" + features["destination_port"].astype(str)
    return features


def _evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int | list[int] | dict[str, float]]:
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
    }


def train_model(db_path: str | Path | None = None, out_dir: str | Path = MODELS_DIR) -> dict[str, object]:
    df = _load_training_data(db_path)
    if "status" in df.columns:
        status_values = df["status"].fillna("").astype(str).str.strip().str.upper()
        valid_statuses = {
            "DELAYED",
            "LATE",
            "LATE_ARRIVAL",
            "ARRIVAL_DELAYED",
            "DELAY",
            "DELIVERED",
            "ON_TIME",
            "ON-TIME",
            "ON TIME",
            "IN_TIME",
            "ARRIVED",
            "ON-TIME-ARRIVAL",
        }
        df = df[status_values.isin(valid_statuses)].copy()

    if "is_delayed_label" not in df.columns:
        df["is_delayed_label"] = build_delay_target(df)
    df = df[df["is_delayed_label"].notna()].copy()

    if len(df) < 3:
        raise ValueError(
            "Not enough real shipment rows to train the delay model. "
            "The repository currently has insufficient real data for a valid booking-time prediction model."
        )

    valid_label_count = int(df["is_delayed_label"].dropna().nunique())
    if valid_label_count < 2:
        raise ValueError(
            "The real training data has only one valid class for is_delayed_label after excluding incomplete arrival rows. "
            "A valid delay model requires both delayed and on-time shipments from actual shipping records."
        )

    train_df, val_df, test_df = time_based_split(df)

    y_train = pd.to_numeric(train_df["is_delayed_label"], errors="coerce").astype("Int64")
    y_val = pd.to_numeric(val_df["is_delayed_label"], errors="coerce").astype("Int64")
    y_test = pd.to_numeric(test_df["is_delayed_label"], errors="coerce").astype("Int64")

    train_df = train_df.drop(columns=[column for column in BANNED_FEATURES if column in train_df.columns], errors="ignore")
    val_df = val_df.drop(columns=[column for column in BANNED_FEATURES if column in val_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[column for column in BANNED_FEATURES if column in test_df.columns], errors="ignore")

    train_mask = y_train.notna()
    val_mask = y_val.notna()
    test_mask = y_test.notna()
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("The split labels still contain missing values after target construction. Please clean or reclassify the shipments before training.")

    train_df = train_df.loc[train_mask].reset_index(drop=True)
    val_df = val_df.loc[val_mask].reset_index(drop=True)
    test_df = test_df.loc[test_mask].reset_index(drop=True)
    y_train = y_train.loc[train_mask].reset_index(drop=True)
    y_val = y_val.loc[val_mask].reset_index(drop=True)
    y_test = y_test.loc[test_mask].reset_index(drop=True)

    X_train = _prepare_model_frame(train_df, history_df=train_df)
    X_val = _prepare_model_frame(val_df, history_df=train_df)
    X_test = _prepare_model_frame(test_df, history_df=train_df)

    y_train = _validate_binary_targets("train", y_train)
    y_val = _validate_binary_targets("validation", y_val)
    y_test = _validate_binary_targets("test", y_test)

    if set(ALLOWED_FEATURES).difference(X_train.columns):
        missing = set(ALLOWED_FEATURES) - set(X_train.columns)
        raise ValueError(f"Training frame is missing required features: {sorted(missing)}")

    model_a = _build_model_pipeline()
    model_b = _build_model_b()
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    val_prob_a = model_a.predict_proba(X_val)[:, 1]
    threshold_a, _ = _score_threshold(y_val, val_prob_a)
    val_prob_b = model_b.predict_proba(X_val)[:, 1]
    threshold_b, _ = _score_threshold(y_val, val_prob_b)

    if model_a is not None:
        test_prob_a = model_a.predict_proba(X_test)[:, 1]
        metrics_a = _evaluate_predictions(y_test, test_prob_a, threshold_a)
        test_prob_b = model_b.predict_proba(X_test)[:, 1]
        metrics_b = _evaluate_predictions(y_test, test_prob_b, threshold_b)

        choose_b = metrics_b["auc"] > metrics_a["auc"] + 0.02 and metrics_b["f1"] > metrics_a["f1"] + 0.02
        chosen_model = model_b if choose_b else model_a
        chosen_threshold = max((threshold_b if choose_b else threshold_a), 0.5)
        chosen_metrics = metrics_b if choose_b else metrics_a
        chosen_model_name = "HistGradientBoostingClassifier" if choose_b else "LogisticRegression"
    else:
        raise RuntimeError("No model was trained.")

    baseline_y = pd.Series(np.ones(len(y_test), dtype=int))
    train_rate = float(pd.to_numeric(y_train, errors="coerce").dropna().mean()) if not y_train.dropna().empty else 0.0
    baseline_prob = np.full(len(y_test), train_rate)
    baseline_predictions = (baseline_prob >= chosen_threshold).astype(int)
    baseline_metrics = {
        "precision": float(precision_score(y_test, baseline_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, baseline_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, baseline_predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_test, baseline_prob)),
        "pr_auc": float(average_precision_score(y_test, baseline_prob)),
        "positive_rate": float(y_test.mean()),
        "threshold": float(chosen_threshold),
        "confusion_matrix": [int(v) for v in confusion_matrix(y_test, baseline_predictions, labels=[0, 1]).ravel()],
    }

    calibration_bins = []
    bin_edges = np.linspace(0.0, 1.0, 11)
    for idx in range(len(bin_edges) - 1):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        mask = (test_prob_a >= lo) & (test_prob_a < hi)
        if idx == len(bin_edges) - 2:
            mask = (test_prob_a >= lo) & (test_prob_a <= hi)
        if mask.any():
            calibration_bins.append(
                {
                    "bin": f"[{lo:.2f},{hi:.2f}]",
                    "count": int(mask.sum()),
                    "avg_predicted_probability": float(test_prob_a[mask].mean()),
                    "observed_rate": float(y_test.iloc[mask].mean()),
                }
            )

    metrics = {
        "model": chosen_model_name,
        "threshold": float(chosen_threshold),
        "precision": float(chosen_metrics["precision"]),
        "recall": float(chosen_metrics["recall"]),
        "f1": float(chosen_metrics["f1"]),
        "auc": float(chosen_metrics["auc"]),
        "pr_auc": float(chosen_metrics["pr_auc"]),
        "positive_rate": float(y_test.mean()),
        "baseline_precision": float(baseline_metrics["precision"]),
        "baseline_recall": float(baseline_metrics["recall"]),
        "baseline_f1": float(baseline_metrics["f1"]),
        "baseline_auc": float(baseline_metrics["auc"]),
        "model_b_precision": float(metrics_b["precision"]),
        "model_b_recall": float(metrics_b["recall"]),
        "model_b_f1": float(metrics_b["f1"]),
        "model_b_auc": float(metrics_b["auc"]),
        "confusion_matrix": [int(v) for v in confusion_matrix(y_test, (test_prob_a >= chosen_threshold).astype(int), labels=[0, 1]).ravel()],
        "calibration_table": calibration_bins,
    }

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": chosen_model,
        "threshold": float(chosen_threshold),
        "version": "1.0",
        "feature_columns": ALLOWED_FEATURES,
        "banned_features": BANNED_FEATURES,
    }
    joblib.dump(artifact, output_dir / "delay_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the booking-time delay model.")
    parser.add_argument("--db", type=str, default="data/curated/supply_chain.duckdb", help="DuckDB or CSV dataset path")
    parser.add_argument("--out", type=str, default="models", help="Output directory for the trained model and metrics")
    args = parser.parse_args()
    print(json.dumps(train_model(db_path=args.db, out_dir=args.out), indent=2))
