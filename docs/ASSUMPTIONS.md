# Assumptions

- Raw data files are available in `data/raw/` before the pipeline runs.
- The platform is designed as an internal operational repository with a small FastAPI service and DuckDB-backed analytics store.
- The delay prediction model uses booking-time features and is a lightweight starting point rather than a production-grade forecasting model.
- LLM access is optional and should be configured via environment variables.
