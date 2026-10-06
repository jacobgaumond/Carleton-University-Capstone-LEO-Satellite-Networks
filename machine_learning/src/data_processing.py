"""Screen ADS-B history data and write a trajectory-only CSV for ML."""

import argparse
from pathlib import Path
from typing import TypedDict

import pandas as pd


# Run this script from the repository root.
DEFAULT_INPUT = "adsb/data/history/adsb_trajectory_data.csv"
DEFAULT_OUTPUT = "machine_learning/data/processed/adsb_trajectory_cleaned.csv"
MAX_ALTITUDE_M = 20_000
MAX_VELOCITY_MPS = 400
MIN_HEADING_DEGREES = 0
MAX_HEADING_DEGREES = 360
OUTPUT_COLUMNS = [
    "time",
    "icao24",
    "lat",
    "lon",
    "geoaltitude",
    "velocity",
    "heading",
]


class ProcessingReport(TypedDict):
    input_file: Path
    input_rows: int
    output_rows: int
    dropped_rows: dict[str, int]
    output_path: Path


def _read_history_file(path: Path) -> pd.DataFrame:
    """Read the trajectory columns needed for processing."""
    data = pd.read_csv(path, usecols=OUTPUT_COLUMNS)
    return data[OUTPUT_COLUMNS]


def _clean_trajectories(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean trajectories and count discarded rows by their first failing rule."""
    cleaned = data[OUTPUT_COLUMNS].copy()
    cleaned["time"] = pd.to_datetime(cleaned["time"], utc=True, errors="coerce")
    cleaned["icao24"] = cleaned["icao24"].astype("string").str.strip()
    for column in ("lat", "lon", "geoaltitude", "velocity", "heading"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Apply checks in order so each rejected row is counted under one reason.
    dropped = {
        "invalid_time_or_aircraft_id": 0,
        "latitude_out_of_range_or_missing": 0,
        "longitude_out_of_range_or_missing": 0,
        "geoaltitude_out_of_range_or_missing": 0,
        "velocity_out_of_range_or_missing": 0,
        "heading_out_of_range_or_missing": 0,
    }

    valid_identity = cleaned["time"].notna() & cleaned["icao24"].notna()
    valid_identity &= cleaned["icao24"].ne("")
    dropped["invalid_time_or_aircraft_id"] = int((~valid_identity).sum())
    cleaned = cleaned.loc[valid_identity]

    latitude_valid = cleaned["lat"].notna() & cleaned["lat"].between(-90, 90)
    dropped["latitude_out_of_range_or_missing"] = int((~latitude_valid).sum())
    cleaned = cleaned.loc[latitude_valid]

    longitude_valid = cleaned["lon"].notna() & cleaned["lon"].between(-180, 180)
    dropped["longitude_out_of_range_or_missing"] = int((~longitude_valid).sum())
    cleaned = cleaned.loc[longitude_valid]

    geoaltitude_valid = cleaned["geoaltitude"].notna() & cleaned[
        "geoaltitude"
    ].between(0, MAX_ALTITUDE_M)
    dropped["geoaltitude_out_of_range_or_missing"] = int(
        (~geoaltitude_valid).sum()
    )
    cleaned = cleaned.loc[geoaltitude_valid]

    velocity_valid = cleaned["velocity"].between(0, MAX_VELOCITY_MPS)
    dropped["velocity_out_of_range_or_missing"] = int((~velocity_valid).sum())
    cleaned = cleaned.loc[velocity_valid]

    heading_valid = cleaned["heading"].between(
        MIN_HEADING_DEGREES, MAX_HEADING_DEGREES
    )
    dropped["heading_out_of_range_or_missing"] = int((~heading_valid).sum())
    cleaned = cleaned.loc[heading_valid]

    return cleaned.reset_index(drop=True), dropped


def process_data(input_location: str, output_location: str) -> ProcessingReport:
    """Process one history CSV and return exact row/column counts for the report."""
    input_file = Path(input_location).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_file}")

    output_path = Path(output_location).resolve()
    if output_path == input_file:
        raise ValueError("The output path must not overwrite an input history file.")

    data = _read_history_file(input_file)
    input_rows = len(data)
    cleaned, dropped_rows = _clean_trajectories(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return {
        "input_file": input_file,
        "input_rows": input_rows,
        "output_rows": len(cleaned),
        "dropped_rows": dropped_rows,
        "output_path": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean ADS-B history CSVs for ML. Run from the repository root."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    report = process_data(args.input, args.output)
    print(f"Read {report['input_rows']} rows from {report['input_file']}.")
    print(f"Wrote {report['output_rows']} rows to {report['output_path']}.")
    print("Rows dropped (each row counted under its first failing reason):")
    for reason, count in report["dropped_rows"].items():
        print(f"  {reason.replace('_', ' ')}: {count}")


if __name__ == "__main__":
    main()
