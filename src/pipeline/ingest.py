from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.common.config import CURATED_DIR, QUARANTINE_DIR, RAW_DIR, REPORTS_DIR
from src.dq.checks import summarize_dataset_quality
from src.dq.report import write_report
from src.pipeline.transform import clean_port_events, clean_ports, clean_shipments


def normalize_for_duckdb(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].astype(object)
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].where(cleaned[column].notna(), None)
    return cleaned


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    conn.execute(schema_path.read_text(encoding="utf-8"))


def run_quality_pipeline() -> dict[str, Any]:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_map = {
        "shipments": {
            "path": RAW_DIR / "shipments.csv",
            "required": ["shipment_id", "origin_port", "destination_port", "booking_date"],
            "transform": clean_shipments,
        },
        "ports": {
            "path": RAW_DIR / "ports.csv",
            "required": ["port_code", "port_name"],
            "transform": clean_ports,
        },
        "port_events": {
            "path": RAW_DIR / "port_events.csv",
            "required": ["event_id", "event_time", "event_type"],
            "transform": clean_port_events,
        },
    }

    raw_reports: dict[str, Any] = {}
    curated_tables: dict[str, pd.DataFrame] = {}
    quarantine_tables: dict[str, pd.DataFrame] = {}
    valid_ports = []

    ports_frame = pd.read_csv(dataset_map["ports"]["path"])
    valid_ports = ports_frame.get("port_code", pd.Series(dtype="object")).astype(str).str.strip().str.upper().tolist()

    for dataset_name, config in dataset_map.items():
        df = pd.read_csv(config["path"])
        raw_reports[dataset_name] = summarize_dataset_quality(
            df,
            dataset_name,
            required_columns=config["required"],
            valid_ports=valid_ports,
        )

        if dataset_name == "shipments":
            curated_df, quarantine_df = config["transform"](df, valid_ports=valid_ports, return_quarantine=True)
        else:
            curated_df, quarantine_df = config["transform"](df, return_quarantine=True)

        curated_tables[dataset_name] = curated_df
        quarantine_tables[dataset_name] = quarantine_df
        if "quarantine_reason" not in quarantine_df.columns and not quarantine_df.empty:
            quarantine_df = quarantine_df.assign(quarantine_reason="unknown")
        curated_path = CURATED_DIR / f"{dataset_name}.csv"
        quarantine_path = QUARANTINE_DIR / f"{dataset_name}_quarantine.csv"
        curated_df.to_csv(curated_path, index=False)
        quarantine_df.to_csv(quarantine_path, index=False)

    db_path = CURATED_DIR / "supply_chain.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)
    for dataset_name, frame in curated_tables.items():
        duckdb_ready = normalize_for_duckdb(frame)
        conn.register(f"{dataset_name}_curated", duckdb_ready)
        conn.execute(f"CREATE OR REPLACE TABLE {dataset_name} AS SELECT * FROM {dataset_name}_curated")
    conn.close()

    curated_reports = {
        name: summarize_dataset_quality(frame, name, required_columns=dataset_map[name]["required"], valid_ports=valid_ports)
        for name, frame in curated_tables.items()
    }

    pipeline_report = {
        "pipeline": "raw_to_curated_quality_gate",
        "raw_quality": raw_reports,
        "curated_quality": curated_reports,
        "quarantine_rows": {name: len(frame) for name, frame in quarantine_tables.items()},
    }
    write_report(pipeline_report, REPORTS_DIR / "pipeline_report.json")
    return pipeline_report


def ingest_all() -> dict[str, int]:
    run_quality_pipeline()
    return {
        "shipments": len(pd.read_csv(CURATED_DIR / "shipments.csv")),
        "ports": len(pd.read_csv(CURATED_DIR / "ports.csv")),
        "port_events": len(pd.read_csv(CURATED_DIR / "port_events.csv")),
    }


if __name__ == "__main__":
    print(run_quality_pipeline())
