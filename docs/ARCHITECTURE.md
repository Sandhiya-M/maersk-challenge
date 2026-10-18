# Architecture

This document describes the intended runtime architecture of the supply-chain platform.

## Layers

1. API layer
   - FastAPI endpoints expose health checks, shipment lookups, route summary, prediction endpoints, and data-quality reporting.
2. Data-quality layer
   - Standalone checks validate CSV shape, nulls, duplicates, and required columns before ingesting raw data.
3. Pipeline layer
   - Raw files are cleaned and loaded into DuckDB as curated tables.
4. ML layer
   - Booking-time features generate a delay model used for prediction and evaluation.
5. Assistant layer
   - An LLM wrapper coordinates prompts, guardrails, and audit logging.
