import pandas as pd

from src.ml.features import BANNED_FEATURES, RouteDelayRateEncoder, build_features


def test_build_features_raises_on_banned_columns() -> None:
    df = pd.DataFrame(
        {
            "shipment_id": ["S1"],
            "booking_date": ["2024-01-01"],
            "origin_port": ["SGSIN"],
            "destination_port": ["USLAX"],
            "cargo_type": ["Electronics"],
            "weight_tons": [1200],
            "container_count": [10],
            "planned_arrival": ["2024-01-10T00:00:00"],
            "actual_arrival": ["2024-01-20T00:00:00"],
            "is_delayed_label": [1],
        }
    )

    try:
        build_features(df)
        raise AssertionError("build_features should reject banned columns")
    except ValueError as exc:
        assert any(banned in str(exc) for banned in BANNED_FEATURES)


def test_threshold_selection_penalizes_false_positive_dominance() -> None:
    from src.ml.train import _score_threshold

    y_true = pd.Series([0, 0, 0, 1, 1])
    probabilities = pd.Series([0.45, 0.44, 0.43, 0.42, 0.41]).to_numpy()

    threshold, _ = _score_threshold(y_true, probabilities)

    assert threshold >= 0.5


def test_route_delay_encoder_uses_training_only_rates() -> None:
    train = pd.DataFrame(
        {
            "route": ["SGSIN->USLAX", "SGSIN->USLAX", "SIN->LAX"],
            "is_delayed_label": [1, 0, 1],
        }
    )
    test = pd.DataFrame({"route": ["SGSIN->USLAX", "SIN->LAX", "NEW->PORT"]})

    encoder = RouteDelayRateEncoder(alpha=5.0)
    encoder.fit(train, train["is_delayed_label"])
    transformed = encoder.transform(test)

    assert transformed["route_delay_rate"].iloc[0] > 0.0
    assert transformed["route_delay_rate"].iloc[2] == encoder.global_rate_


def test_build_delay_target_uses_status_fallback_when_dates_are_missing() -> None:
    from src.ml.features import build_delay_target

    df = pd.DataFrame(
        {
            "planned_arrival": ["2024-01-10", "2024-01-10", "2024-01-10"],
            "actual_arrival": [None, None, None],
            "status": ["DELAYED", "DELIVERED", "CANCELLED"],
        }
    )

    target = build_delay_target(df)
    assert target.iloc[0] == 1
    assert target.iloc[1] == 0
    assert pd.isna(target.iloc[2])


def test_time_split_keeps_train_before_test() -> None:
    from src.ml.train import time_based_split

    df = pd.DataFrame(
        {
            "booking_date": pd.to_datetime([
                "2024-01-01",
                "2024-01-05",
                "2024-01-10",
                "2024-01-15",
                "2024-01-20",
                "2024-01-25",
                "2024-01-30",
                "2024-02-05",
            ]),
            "is_delayed_label": [0, 0, 1, 0, 1, 0, 1, 1],
        }
    )

    train, val, test = time_based_split(df)
    assert train["booking_date"].max() <= test["booking_date"].min()
    assert val["booking_date"].min() >= train["booking_date"].min()
    assert val["booking_date"].max() <= test["booking_date"].min()


def test_prediction_returns_valid_probability_and_keys() -> None:
    from src.ml.predict import predict_delay

    payload = {
        "shipment_id": "SHP-001",
        "booking_date": "2024-02-05",
        "origin_port": "SGSIN",
        "destination_port": "USLAX",
        "cargo_type": "Electronics",
        "weight_tons": 1200,
        "container_count": 10,
        "planned_arrival": "2024-02-10T00:00:00",
        "planned_departure": "2024-02-08T00:00:00",
    }

    out = predict_delay(payload)
    assert set(out) >= {"shipment_id", "is_more_than_24h_late", "probability_gt_24h_late", "predicted_delay_hours"}
    assert 0.0 <= out["probability_gt_24h_late"] <= 1.0
