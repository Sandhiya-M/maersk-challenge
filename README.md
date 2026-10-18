# Supply Chain Platform

A lightweight supply-chain analytics platform for shipment monitoring, route analysis, data-quality reporting, forecasted delay risk, and natural-language queries over the curated data set.

## Overview

The application combines a FastAPI backend, DuckDB-backed data pipeline, a booking-time delay prediction model, and a tool-calling assistant that talks to a local Ollama model. The system is designed to answer questions such as:

- Which routes have the highest average delay?
- What is the on-time performance for a route?
- Which shipments are delayed or at risk?
- What does the current data-quality status look like?

The platform is intentionally local-first and does not depend on OpenAI or any external API key. The assistant uses local Ollama and the project tool registry to invoke real data queries instead of guessing from memory.

## Features

- Shipment, port, and port-event data ingestion and querying
- Route summary and delay analysis over curated data
- Data-quality validation and reporting
- Delay prediction using a booking-time model
- Local LLM assistant with tool calling and audit logging
- FastAPI endpoints for operational and ML workflows
- Dockerized deployment for local setups

## Architecture

The platform is structured into these runtime layers:

1. API layer
   - FastAPI routes expose health, shipment, route, prediction, and assistant endpoints.
2. Data-quality layer
   - CSV validation and reporting identify nulls, duplicates, required columns, and schema issues.
3. Pipeline layer
   - Raw data is transformed and ingested into DuckDB as curated tables.
4. ML layer
   - Booking-time features feed a delay model for risk scoring and evaluation.
5. Assistant layer
   - A tool-calling wrapper sends requests to Ollama and executes dataset queries before composing the answer.

## Project structure

```text
supply-chain-platform/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .env
├── .env.example
├── data/
│   ├── raw/
│   ├── curated/
│   └── quarantine/
├── docs/
├── logs/
├── models/
├── reports/
├── scripts/
├── src/
├── tests/
├── ui/
└── frontend/
```

## Prerequisites

- Python 3.11+
- Ollama installed locally
- An Ollama model available, for example `qwen2.5:3b` or `qwen2.5:7b-instruct`
- Docker and Docker Compose (optional for containerized runs)

## Local setup

1. Start Ollama.
2. Pull a model:

```bash
ollama pull qwen2.5:3b
```

Or use the larger instruct variant if you prefer:

```bash
ollama pull qwen2.5:7b-instruct
```

3. Create the environment file, or copy the sample:

```bash
cp .env.example .env
```

4. Install Python dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

5. Run the backend:

```bash
make run
```

The API will be available at:

- http://localhost:8000
- Health check: http://localhost:8000/health

## Assistant setup

The assistant expects the Ollama provider and a local model:

```env
LLM_PROVIDER=ollama
LLM_API_KEY=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
LLM_MODEL=qwen2.5:3b
```

The project intentionally does not use OpenAI or OpenAPI endpoints for the active runtime path.

To ask the assistant a question from the CLI:

```bash
python -m src.assistant.cli --query "Which route has the highest average delay this quarter?"
```

Or use the interactive shell:

```bash
python -m src.assistant.cli --interactive
```

Windows PowerShell wrappers are also included:

```powershell
./scripts/assistant.cmd --query "Which route has the highest average delay this quarter?"
./scripts/assistant.cmd --interactive
```

## Docker Compose

The project includes a Docker Compose file that builds and runs the FastAPI app container. The service starts the app on port 8000.

```bash
docker compose up --build
```

Then open:

- http://localhost:8000

Important note for Compose:

- Inside a container, `localhost` refers to the container itself, not the host machine.
- For Docker Desktop or host-based Ollama installs, the app must reach the host via `host.docker.internal`.
- The Compose file therefore sets the Ollama base URL to `http://host.docker.internal:11434` for the app container.
- If you run the app directly on the host OS, keep `OLLAMA_BASE_URL=http://localhost:11434` instead.

If Ollama is not running, or the model is missing, the assistant will fail with a clear error. Pull the model first:

```bash
ollama pull qwen2.5:3b
```

## Data model

The project is built around curated shipping data:

- Shipment
  - shipment_id
  - booking_time
  - port_of_loading
  - port_of_discharge
  - cargo_type
  - vessel_name
  - delay_hours
  - shipment_status
- Port
  - port_code
  - port_name
  - country
  - region
- Port event
  - event_id
  - port_code
  - event_time
  - event_type

## Data-quality and ML workflows

The project includes a data-quality stage and a route-delay prediction model.

Common commands:

```bash
make dq
make ingest
make train
make test
```

The app also exposes endpoints for:

- `/health`
- `/shipments`
- `/ports`
- `/port-events`
- `/routes`
- `/predict`
- `/predict-delay`
- `/assistant/chat`
- `/data-quality`

## Guardrails and auditability

The assistant enforces a narrow operational scope:

- It only answers supply-chain questions about shipments, routes, ports, delay risk, and logistics performance.
- It calls tools before answering factual questions.
- It logs calls and responses to `logs/llm_audit.jsonl`.
- It avoids guesswork and tries to cite the tool used for the result.

## Troubleshooting

### Ollama is unreachable

Check the local server:

```bash
curl http://localhost:11434/api/tags
```

If the app is running in Docker, use `host.docker.internal` instead of `localhost` for `OLLAMA_BASE_URL`.

### Wrong endpoint

The assistant uses the native Ollama chat API:

```text
/api/chat
```

It does not use an OpenAI-compatible `/v1` endpoint for the local runtime path.

### Model missing

Pull the model explicitly:

```bash
ollama pull qwen2.5:3b
```

Then confirm the configured model matches the pulled model name.

## Submission checklist

- Local Ollama-only configuration is used
- No OpenAI/OpenAPI runtime path is required for normal operation
- App boots via FastAPI on port 8000
- Data-quality and prediction workflows are available
- Assistant uses tool calling against the local dataset
- Docker Compose is documented for local deployment

---

This repository is intended to be a usable local analytics platform and a starting point for operational supply-chain monitoring, delay prediction, and assistant-driven exploration.
