.PHONY: run test lint dq ingest train assistant

run:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

assistant:
	python -m src.assistant.cli --interactive

test:
	pytest -q

lint:
	ruff check src tests

# Run standalone data quality checks.
dq:
	python -m src.dq.dq_check

# Load raw-to-curated data into DuckDB.
ingest:
	python -m src.pipeline.ingest

# Train the delay prediction model.
train:
	python -m src.ml.train
