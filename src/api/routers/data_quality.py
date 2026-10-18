from __future__ import annotations

import json

from fastapi import APIRouter

from src.common.config import REPORTS_DIR

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


@router.get("")
def get_data_quality_report() -> dict[str, object]:
    report_paths = [
        REPORTS_DIR / "pipeline_report.json",
        REPORTS_DIR / "dq_report.json",
    ]

    for report_path in report_paths:
        if report_path.exists():
            return {
                "status": "ok",
                "report": json.loads(report_path.read_text(encoding="utf-8")),
            }

    return {"status": "not_run", "checks": []}
