# Supply Chain Platform

A local-first supply-chain analytics platform for shipment monitoring, route analysis, data-quality reporting, ML-based delay prediction, and natural-language querying over curated logistics data.

## Overview

This project combines:

- A FastAPI backend for shipment, route, quality, and prediction APIs
- A DuckDB-backed ingestion pipeline for curated logistics data
- A booking-time delay prediction model trained on real freight records
- A local LLM assistant powered by Ollama that can query project data instead of guessing
- A React + Vite frontend for operational dashboards and quick inspection

The platform is designed to answer questions such as:

- Which routes have the highest delay risk?
- What is the on-time performance by port or route?
- Which shipments are currently delayed or likely to be delayed?
- What is the current data-quality status across the dataset?

This is intentionally a local-first solution. It does not depend on external SaaS APIs for its active runtime path and can run fully on a developer machine with local dependencies.

## Key Features

- Shipment, port, and port-event ingestion and querying
- Route and delay analysis over curated shipping data
- Data-quality checks and reports for missing values, duplicates, and schema issues
- Booking-time delay prediction using a trained ML model
- Assistant tooling for natural-language access to the data model
- FastAPI endpoints for operational and diagnostic workflows
- Docker support for local containerized development
- A Vite/React frontend served on a separate port

## Architecture

The platform is structured around several runtime layers:

1. API layer
   - FastAPI routes expose health, shipment, route, prediction, and assistant endpoints.
2. Data-quality layer
   - Validation and reporting identify nulls, duplicates, required columns, and other quality issues.
3. Pipeline layer
   - Raw CSV data is processed and loaded into DuckDB-backed curated tables.
4. ML layer
   - Booking-time features feed a delay model and evaluation pipeline.
5. Assistant layer
   - The local LLM wrapper calls project tooling and executes dataset queries before answering.

## Repository Layout

```text
supply-chain-platform/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   ├── curated/
│   └── quarantine/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ASSUMPTIONS.md
│   ├── DATA_MODEL.md
│   ├── DQ_DECISIONS.md
│   └── ML_REPORT.md
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
├── logs/
├── models/
├── reports/
├── scripts/
│   ├── assistant.cmd
│   ├── assistant.ps1
│   └── entrypoint.sh
├── src/
│   ├── api/
│   ├── assistant/
│   ├── common/
│   ├── dq/
│   ├── ml/
│   ├── pipeline/
│   └── __init__.py
├── tests/
│   ├── integration/
│   ├── unit/
│   └── conftest.py
├── ui/
└── .env
```

## Prerequisites

Before running the platform, install the following:

- Python 3.11+
- Node.js 20+
- npm
- Ollama installed locally
- A local Ollama model, such as `qwen2.5:3b` or `qwen2.5:7b-instruct`
- Docker Desktop + Docker Compose if you want to run the project in containers

## Environment Configuration

Create a local environment file from the sample:

```bash
cp .env.example .env
```

The default environment variables look like this:

```env
LLM_PROVIDER=ollama
LLM_API_KEY=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
LLM_MODEL=qwen2.5:3b
APP_ENV=development
```

If you are running the services in Docker Compose, use `host.docker.internal` instead of `localhost` for the Ollama base URL inside the container. The compose file is set up for that pattern.

## Local Setup

### 1) Start Ollama and pull a model

```bash
ollama pull qwen2.5:3b
```

You can also use the larger instruct variant:

```bash
ollama pull qwen2.5:7b-instruct
```

### 2) Create a virtual environment and install Python dependencies

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 3) Install frontend dependencies

```bash
cd frontend
npm install
```

## Run the Backend

The project includes a Makefile for common developer tasks.

```bash
make run
```

This starts the FastAPI app with Uvicorn on:

- http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

You can also start it directly:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Run the Frontend

From the project root:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Then open:

```text
http://localhost:5173
```

The frontend expects the backend to be available at the default API location the project config uses.

## Run the Full Stack with Docker Compose

From the project root:

```bash
docker compose up --build
```

This starts both services:

- Backend: http://localhost:8000
- Frontend: http://localhost:5173

### Docker notes

- Docker Desktop must be running before `docker compose` works.
- The compose file includes a frontend service and wires it to the app container.
- For host-based Ollama, the compose environment uses `http://host.docker.internal:11434` so the app container can connect to the host machine.
- If Docker is not installed or the daemon is not running, Compose will fail with a Docker API connection error.

## Useful Project Commands

These commands are defined in the Makefile:

```bash
make run
make dq
make ingest
make train
make test
make lint
make assistant
```

### Data quality

```bash
make dq
```

### Load raw data into DuckDB

```bash
make ingest
```

### Train the delay model

```bash
make train
```

### Run tests

```bash
make test
```

### Lint the codebase

```bash
make lint
```

### Start the interactive assistant shell

```bash
make assistant
```

## Assistant Usage

The project includes a local assistant CLI that calls tools against the platform data and can answer operational questions using a local LLM.

From the project root:

```bash
python -m src.assistant.cli --query "Which route has the highest average delay this quarter?"
```

Interactive mode:

```bash
python -m src.assistant.cli --interactive
```

PowerShell wrappers are also included:

```powershell
./scripts/assistant.cmd --query "Which route has the highest average delay this quarter?"
./scripts/assistant.cmd --interactive
```

## API Overview

The backend exposes these main areas:

- `/` - service greeting
- `/health` - health/status information
- `/shipments` - shipment queries
- `/ports` - port metadata and summaries
- `/port-events` - port event queries
- `/routes` - route-level analytics
- `/predict` - delay prediction endpoint
- `/assistant` - LLM assistant interaction endpoints
- `/dq` - data-quality endpoints
- `/pipeline` - ingestion and pipeline actions

The API also exposes Swagger/OpenAPI docs at:

- http://localhost:8000/docs

## Data Model

The platform relies on curated shipping data mainly from these entity sets:

### Shipment
- `shipment_id`
- `booking_time`
- `port_of_loading`
- `port_of_discharge`
- `cargo_type`
- `vessel_name`
- `delay_hours`
- `shipment_status`

### Port
- `port_code`
- `port_name`
- `country`
- `region`

### Port Event
- `event_id`
- `port_code`
- `event_time`
- `event_type`

## ML and Data Quality Workflow

The project includes both upstream data validation and downstream predictive analytics:

```bash
make dq
make ingest
make train
make test
```

The training pipeline loads real data, builds booking-time features, fits the delay model, evaluates thresholds, and writes artifacts into the `models/` directory. Data quality reports are generated in the `reports/` folder.

## Documentation

Additional project docs live in the `docs/` directory:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)
- [docs/DATA_MODEL.md](docs/DATA_MODEL.md)
- [docs/DQ_DECISIONS.md](docs/DQ_DECISIONS.md)
- [docs/ML_REPORT.md](docs/ML_REPORT.md)

## Troubleshooting

### Ollama errors

If the assistant or backend cannot connect to Ollama:

```bash
ollama ps
ollama pull qwen2.5:3b
```

Verify the environment variable `OLLAMA_BASE_URL` matches your setup.

### Docker connection errors

If Compose fails with a Docker daemon connection error, start Docker Desktop and confirm the engine is running:

```bash
docker ps
```

### Frontend not loading

Check that the Vite app is started in the frontend folder and that the backend is running on port 8000.

## License

This project is intended for local development and internal analytics workflows. Update or remove this section if you are preparing a production or distribution package.

## Summary

This repository is a working local analytics platform with:

- Python backend
- DuckDB data workflows
- model training/evaluation
- React frontend
- local Ollama-based assistant
- Docker Compose support

It is ready for developer setup, experimentation, and local runs with the correct environment and dependencies installed.
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
