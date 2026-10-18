from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_dq_report(results: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    passed = all(value.get("passed", False) for value in results.values())
    return {
        "dataset": dataset_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": results,
    }


def write_report(report: dict[str, Any], target_path: str | Path) -> Path:
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
