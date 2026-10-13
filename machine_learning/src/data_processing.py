"""Screen ADS-B history data and write a trajectory-only CSV for ML."""

import argparse
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


# Run this script from the repository root.
DEFAULT_INPUT = "adsb/data/history/adsb_trajectory_data.csv"
DEFAULT_OUTPUT = "machine_learning/data/processed/adsb_trajectory_cleaned.csv"
MAX_ALTITUDE_M = 30_000
MAX_VELOCITY_MPS = 400
MAX_POSITION_CHANGE_MPS = 400
MAX_VELOCITY_CHANGE_MPS_PER_SECOND = 10
MAX_ALTITUDE_CHANGE_MPS = 30
DYNAMIC_CHECK_INTERVAL_SECONDS = 10
MIN_HEADING_DEGREES = 0
MAX_HEADING_DEGREES = 360
EARTH_RADIUS_M = 6_371_000
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
    remaining_flight_paths: int
    dropped_rows: dict[str, int]
    output_path: Path


def _read_history_file(path: Path) -> pd.DataFrame:
    """Read the trajectory columns needed for processing."""
    data = pd.read_csv(path, usecols=OUTPUT_COLUMNS)
    return data[OUTPUT_COLUMNS]


def _clean_trajectories(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean trajectories and count discarded rows by their first failing rule."""
    cleaned = data[OUTPUT_COLUMNS].copy()

    # Parse timestamps and numeric measurements; invalid values become missing
    # so the validation checks below can reject them consistently.
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
        "incomplete_flight_path": 0,
        "position_change_exceeds_threshold": 0,
        "velocity_change_exceeds_threshold": 0,
        "altitude_change_exceeds_threshold": 0,
    }

    # Require a valid timestamp and a non-empty aircraft identifier before
    # grouping rows into flight paths.
    valid_identity = cleaned["time"].notna() & cleaned["icao24"].notna()
    valid_identity &= cleaned["icao24"].ne("")
    dropped["invalid_time_or_aircraft_id"] = int((~valid_identity).sum())
    cleaned = cleaned.loc[valid_identity]

    # Keep only geographic latitudes within the valid range.
    latitude_valid = cleaned["lat"].notna() & cleaned["lat"].between(-90, 90)
    dropped["latitude_out_of_range_or_missing"] = int((~latitude_valid).sum())
    cleaned = cleaned.loc[latitude_valid]

    # Keep only geographic longitudes within the valid range.
    longitude_valid = cleaned["lon"].notna() & cleaned["lon"].between(-180, 180)
    dropped["longitude_out_of_range_or_missing"] = int((~longitude_valid).sum())
    cleaned = cleaned.loc[longitude_valid]

    # Require altitude to be present and between ground level and the configured
    # maximum flight altitude.
    geoaltitude_valid = cleaned["geoaltitude"].notna() & cleaned[
        "geoaltitude"
    ].between(0, MAX_ALTITUDE_M)
    dropped["geoaltitude_out_of_range_or_missing"] = int(
        (~geoaltitude_valid).sum()
    )
    cleaned = cleaned.loc[geoaltitude_valid]

    # Require a present, non-negative velocity no greater than the configured
    # maximum.
    velocity_valid = cleaned["velocity"].between(0, MAX_VELOCITY_MPS)
    dropped["velocity_out_of_range_or_missing"] = int((~velocity_valid).sum())
    cleaned = cleaned.loc[velocity_valid]

    # Keep headings within the inclusive compass range.
    heading_valid = cleaned["heading"].between(
        MIN_HEADING_DEGREES, MAX_HEADING_DEGREES
    )
    dropped["heading_out_of_range_or_missing"] = int((~heading_valid).sum())
    cleaned = cleaned.loc[heading_valid]

    # Check timestamp continuity in chronological order for each aircraft.
    # If any adjacent samples are not exactly one second apart, reject every
    # row in that flight path rather than leaving a partial trajectory.
    ordered = cleaned.sort_values(["icao24", "time"])
    time_diffs = ordered.groupby("icao24", sort=False)["time"].diff()
    incomplete_icao24 = ordered.loc[
        time_diffs.notna() & time_diffs.ne(pd.Timedelta(seconds=1)), "icao24"
    ].unique()
    incomplete_path = cleaned["icao24"].isin(incomplete_icao24)
    dropped["incomplete_flight_path"] = int(incomplete_path.sum())
    cleaned = cleaned.loc[~incomplete_path]

    # Sort and group the remaining complete paths so dynamic changes can be
    # compared over each ten-second interval for the same aircraft.
    ordered = cleaned.sort_values(["icao24", "time"])
    grouped = ordered.groupby("icao24", sort=False)
    interval_time = grouped["time"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS)
    elapsed_seconds = (ordered["time"] - interval_time).dt.total_seconds()

    # Compare each point with the position ten seconds earlier. Convert
    # coordinates to radians and use haversine distance over Earth's surface.
    previous_latitude = np.radians(
        grouped["lat"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS)
    )
    previous_longitude = np.radians(
        grouped["lon"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS)
    )
    latitude = np.radians(ordered["lat"])
    longitude = np.radians(ordered["lon"])
    delta_latitude = latitude - previous_latitude
    delta_longitude = longitude - previous_longitude
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(previous_latitude)
        * np.cos(latitude)
        * np.sin(delta_longitude / 2) ** 2
    )
    position_change_mps = (
        2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    ) / elapsed_seconds

    # Reject a path if its average ground speed over any ten-second interval
    # exceeds the allowed position-change rate.
    position_violation = position_change_mps > MAX_POSITION_CHANGE_MPS

    # Compare reported velocities ten seconds apart, normalized by elapsed
    # time, to find implausible average changes in speed.
    velocity_change_mps_per_second = (
        grouped["velocity"].diff(DYNAMIC_CHECK_INTERVAL_SECONDS).abs()
        / elapsed_seconds
    )
    velocity_violation = (
        velocity_change_mps_per_second
        > MAX_VELOCITY_CHANGE_MPS_PER_SECOND
    )

    # Compare altitudes ten seconds apart, normalized by elapsed time, to find
    # implausible average vertical rates of change.
    altitude_change_mps = (
        grouped["geoaltitude"].diff(DYNAMIC_CHECK_INTERVAL_SECONDS).abs()
        / elapsed_seconds
    )
    altitude_violation = altitude_change_mps > MAX_ALTITUDE_CHANGE_MPS

    # Identify paths with a violation of each dynamic rule. If a path violates
    # multiple rules, count it under its first failing rule so drop counts stay
    # exclusive and sum to the number of rows removed by dynamic filtering.
    position_icao24 = ordered.loc[position_violation, "icao24"].unique()
    velocity_icao24 = ordered.loc[velocity_violation, "icao24"].unique()
    altitude_icao24 = ordered.loc[altitude_violation, "icao24"].unique()

    position_path = cleaned["icao24"].isin(position_icao24)
    velocity_path = cleaned["icao24"].isin(velocity_icao24) & ~position_path
    altitude_path = (
        cleaned["icao24"].isin(altitude_icao24)
        & ~position_path
        & ~velocity_path
    )
    dropped["position_change_exceeds_threshold"] = int(position_path.sum())
    dropped["velocity_change_exceeds_threshold"] = int(velocity_path.sum())
    dropped["altitude_change_exceeds_threshold"] = int(altitude_path.sum())

    dynamic_path = position_path | velocity_path | altitude_path
    cleaned = cleaned.loc[~dynamic_path]

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
        "remaining_flight_paths": int(cleaned["icao24"].nunique()),
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
    print(f"\nWrote {report['output_rows']} rows to {report['output_path']}.")
    print(f"\nRemaining full flight paths: {report['remaining_flight_paths']}.")
    print("\nRows dropped (each row counted under its first failing reason):")
    for reason, count in report["dropped_rows"].items():
        print(f"  {reason.replace('_', ' ')}: {count}")


if __name__ == "__main__":
    main()
