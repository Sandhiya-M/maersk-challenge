#!/usr/bin/env bash
set -euo pipefail

python -m src.dq.dq_check
python -m src.pipeline.ingest
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
