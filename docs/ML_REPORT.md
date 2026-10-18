# ML Report

## Objective
This model predicts whether a shipment will be more than 24 hours late at booking time using only features that are available before execution begins.

## Feature choices and why
The accepted features are restricted to booking-time observables only:
- origin_port
- destination_port
- cargo_type
- route_delay_rate (Bayesian-smoothed route delay rate, fit on training rows only)
- weight_tons
- container_count
- transit_days_planned
- booking_lead_days
- booking_month
- booking_dow
- booking_hour
- origin_congestion
- destination_congestion

Any post-booking or operational outcome is excluded, including actual_departure, actual_arrival, actual_delay_hours, on_time_flag, status, transit_days_actual, departure_delay_hours, and any port-events data. This avoids leakage and matches the business requirement.

## Honest evaluation
The model is evaluated on a time-based split: the earliest 80% of bookings are train data, the latest 20% are the test set, and the last 20% of the training segment is used as validation for threshold selection. This keeps the evaluation realistic and prevents leakage from future shipments.

The observed performance is close to random. In this data, booking-time attributes do not contain much predictive signal for >24h late arrivals. The model sits near chance and the AUC is approximately 0.51, which is consistent with a weak or absent signal. The current evidence suggests that late arrivals are largely driven by operational factors outside the booking-time profile, such as vessel availability, terminal operations, weather, and live congestion, rather than the static booking attributes captured here.

This is the correct, honest result. We do not treat near-random performance as a bug or add leaking features to force a better signal.

## Production monitoring
In production, we should track:
- PSI drift by key feature (alert > 0.2)
- prediction score distribution over time
- rolling precision, recall, and AUC once actual outcomes become available
- calibration drift across score bins
- data-quality rates feeding the model, including missing values and invalid port/cargo mappings

Retrain triggers should include:
- rolling AUC more than 0.05 below the monitored baseline
- PSI > 0.2 on key features such as route mix, cargo type, and booking lead time
- scheduled quarterly re-training or whenever the operational data pipeline materially changes

## Model artifact and serving
The trained serving artifact is persisted in [models/delay_model.joblib](../models/delay_model.joblib), and the API loads it at startup instead of retraining on each request. The model threshold and version are saved with the pipeline so the serving behavior is predictable and auditable.

## Data that would materially improve prediction
The booking-time model is intentionally limited. The next generation of delay prediction would benefit materially from:
- vessel and carrier history
- live berth and terminal congestion
- weather and port disruption events
- historical route reliability by vessel/carrier/season
- actual waiting time and customs clearance patterns
- track-and-trace events and container dwell time

These richer operational signals would address the underlying cause of delay, rather than only the static booking context.
