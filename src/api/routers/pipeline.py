from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, UploadFile

from src.common.config import RAW_DIR
from src.pipeline.ingest import run_quality_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/upload")
async def upload_csvs(
    shipments: UploadFile | None = File(default=None),
    ports: UploadFile | None = File(default=None),
    port_events: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for key, file in {"shipments": shipments, "ports": ports, "port_events": port_events}.items():
        if file is None or not file.filename:
            continue
        target_name = {
            "shipments": "shipments.csv",
            "ports": "ports.csv",
            "port_events": "port_events.csv",
        }[key]
        target_path = RAW_DIR / target_name
        target_path.write_bytes(await file.read())
        saved[key] = file.filename

    return {
        "status": "ok",
        "saved": saved,
        "message": "CSV files were stored in the raw ingestion folder.",
    }


@router.post("/run")
def run_pipeline() -> dict[str, Any]:
    report = run_quality_pipeline()
    return {
        "status": "ok",
        "pipeline": "raw_to_curated_quality_gate",
        "message": "Pipeline completed successfully.",
        "report": report,
        "raw_quality": report.get("raw_quality", {}),
        "curated_quality": report.get("curated_quality", {}),
        "quarantine_rows": report.get("quarantine_rows", {}),
    }
