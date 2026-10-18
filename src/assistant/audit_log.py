from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import LOGS_DIR


def append_audit_entry(
    prompt: str,
    response: str | None = None,
    model: str = "assistant-local",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    final_output: str | None = None,
    session_id: str | None = None,
) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "llm_audit.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "session_id": session_id,
        "prompt": prompt,
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "final_output": final_output or response or "",
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
    return log_path
