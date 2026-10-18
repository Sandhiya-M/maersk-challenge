from __future__ import annotations

from pathlib import Path

import duckdb

from src.common.config import CURATED_DIR

DB_PATH = CURATED_DIR / "supply_chain.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))
