from __future__ import annotations

import argparse
import json

import pandas as pd

from src.common.config import RAW_DIR, REPORTS_DIR
from src.dq.checks import summarize_dataset_quality
from src.dq.report import write_report


def evaluate_dataset(path: str, required_columns: list[str], valid_ports: list[str] | None = None) -> dict[str, object]:
    df = pd.read_csv(path)
    return summarize_dataset_quality(df, path.split("/")[-1].split(".")[0], required_columns, valid_ports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone data-quality checker")
    parser.add_argument("--dataset", default="shipments", help="Dataset key to validate")
    parser.add_argument("--curated", action="store_true", help="Validate the curated data instead of raw data")
    args = parser.parse_args()

    dataset_map = {
        "shipments": ["shipment_id", "origin_port", "destination_port", "booking_date"],
        "ports": ["port_code", "port_name"],
        "port_events": ["event_id", "event_time", "event_type"],
    }
    target = dataset_map.get(args.dataset, dataset_map["shipments"])
    csv_path = (RAW_DIR / f"{args.dataset}.csv") if not args.curated else (RAW_DIR / f"{args.dataset}.csv")
    valid_ports = []
    if args.dataset == "shipments":
        ports_df = pd.read_csv(RAW_DIR / "ports.csv")
        valid_ports = ports_df["port_code"].astype(str).str.strip().str.upper().tolist()

    report = evaluate_dataset(str(csv_path), target, valid_ports)
    output_path = REPORTS_DIR / (f"{args.dataset}_curated_report.json" if args.curated else f"{args.dataset}_dq_report.json")
    write_report(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
