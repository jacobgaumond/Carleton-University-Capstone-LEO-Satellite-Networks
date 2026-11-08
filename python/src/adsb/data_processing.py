"""Screen ADS-B history data and write a trajectory-only CSV for ML."""

import argparse
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Run this script from the repository root.
ROOT_DIR_PATH = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT = ROOT_DIR_PATH / "out" / "adsb/data/history/adsb_trajectory_data.csv"
DEFAULT_OUTPUT = ROOT_DIR_PATH / "out" / "machine_learning/data/processed/adsb_trajectory_cleaned.csv"
MAX_ALTITUDE_M = 30_000
MAX_VELOCITY_MPS = 400
MAX_DISPLACEMENT_ERROR_FRACTION = 0.25
MAX_POSITION_ERROR_M = 50
MAX_VELOCITY_CHANGE_MPS_PER_SECOND = 10
MAX_ALTITUDE_CHANGE_MPS = 30
SAMPLE_INTERVAL_SECONDS = 1
DYNAMIC_CHECK_INTERVAL_SECONDS = 10
MIN_FLIGHT_GAP_SECONDS = 5
MIN_FLIGHT_DURATION_MINUTES = 10
FLIGHT_PATH_OUTPUT_COLUMN = "flight_path_id"
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
    starting_flight_paths: int
    output_rows: int
    remaining_flight_paths: int
    flight_path_splits_at_start: int
    flight_path_splits_at_end: int
    dropped_rows: dict[str, int]
    output_path: Path


class FlightPathSplitReport(TypedDict):
    starting_flight_paths: int
    remaining_flight_paths: int
    flight_path_splits_at_start: int
    flight_path_splits_at_end: int


def _read_history_file(path: Path) -> pd.DataFrame:
    """Read the trajectory columns needed for processing."""
    data = pd.read_csv(path, usecols=OUTPUT_COLUMNS)
    return data[OUTPUT_COLUMNS]


def _filter_missing_values(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with missing trajectory data or an empty aircraft ID."""
    missing = data[OUTPUT_COLUMNS].isna().any(axis=1)
    missing |= data["icao24"].eq("")
    return data.loc[~missing], int(missing.sum())


def _filter_out_of_range_values(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with measurements outside their valid ranges."""
    out_of_range = (
        (data["lat"].notna() & ~data["lat"].between(-90, 90))
        | (data["lon"].notna() & ~data["lon"].between(-180, 180))
        | (
            data["geoaltitude"].notna()
            & ~data["geoaltitude"].between(0, MAX_ALTITUDE_M)
        )
        | (
            data["velocity"].notna()
            & ~data["velocity"].between(0, MAX_VELOCITY_MPS)
        )
        | (
            data["heading"].notna()
            & ~data["heading"].between(MIN_HEADING_DEGREES, MAX_HEADING_DEGREES)
        )
    )
    return data.loc[~out_of_range], int(out_of_range.sum())


def _split_flight_paths(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Assign a new flight path after each sufficiently long timestamp gap."""
    ordered = data.sort_values(["icao24", "time"])
    time_diffs = ordered.groupby("icao24", sort=False)["time"].diff()
    split_boundary = time_diffs.ge(
        pd.Timedelta(seconds=MIN_FLIGHT_GAP_SECONDS)
    )
    new_path = time_diffs.isna() | split_boundary
    return (
        ordered.assign(**{FLIGHT_PATH_OUTPUT_COLUMN: new_path.cumsum()}),
        int(split_boundary.sum()),
    )


def _filter_short_flight_paths(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove aircraft paths shorter than the minimum duration."""
    flight_times = data.groupby(FLIGHT_PATH_OUTPUT_COLUMN, sort=False)["time"].agg(
        ["min", "max"]
    )
    short_paths = flight_times.index[
        flight_times["max"] - flight_times["min"]
        < pd.Timedelta(minutes=MIN_FLIGHT_DURATION_MINUTES)
    ]
    short_path = data[FLIGHT_PATH_OUTPUT_COLUMN].isin(short_paths)
    return data.loc[~short_path], int(short_path.sum())


def _east_north_displacement(
    latitude_1: NDArray[np.float64],
    longitude_1: NDArray[np.float64],
    latitude_2: NDArray[np.float64],
    longitude_2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return local east and north displacements in meters."""
    latitude_1_rad = np.radians(latitude_1)
    longitude_1_rad = np.radians(longitude_1)
    latitude_2_rad = np.radians(latitude_2)
    longitude_2_rad = np.radians(longitude_2)
    delta_latitude = latitude_2_rad - latitude_1_rad
    delta_longitude = (
        (longitude_2_rad - longitude_1_rad + np.pi) % (2 * np.pi)
    ) - np.pi
    mean_latitude = (latitude_1_rad + latitude_2_rad) / 2
    east_displacement_m = (
        EARTH_RADIUS_M * np.cos(mean_latitude) * delta_longitude
    )
    north_displacement_m = EARTH_RADIUS_M * delta_latitude
    return east_displacement_m, north_displacement_m


def _reported_east_north_displacement(
    velocity_mps: NDArray[np.float64],
    heading_degrees: NDArray[np.float64],
    flight_path_ids: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return reported displacement over the configured comparison interval."""
    heading_radians = np.radians(heading_degrees)
    components = pd.DataFrame(
        {
            FLIGHT_PATH_OUTPUT_COLUMN: flight_path_ids,
            "velocity_east": velocity_mps * np.sin(heading_radians),
            "velocity_north": velocity_mps * np.cos(heading_radians),
        }
    )
    grouped = components.groupby(FLIGHT_PATH_OUTPUT_COLUMN, sort=False)
    east_segments = grouped["velocity_east"].shift(SAMPLE_INTERVAL_SECONDS)
    north_segments = grouped["velocity_north"].shift(SAMPLE_INTERVAL_SECONDS)
    east_displacement = (
        east_segments.groupby(components[FLIGHT_PATH_OUTPUT_COLUMN], sort=False)
        .rolling(
            window=DYNAMIC_CHECK_INTERVAL_SECONDS,
            min_periods=DYNAMIC_CHECK_INTERVAL_SECONDS,
        )
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(components.index)
    )
    north_displacement = (
        north_segments.groupby(components[FLIGHT_PATH_OUTPUT_COLUMN], sort=False)
        .rolling(
            window=DYNAMIC_CHECK_INTERVAL_SECONDS,
            min_periods=DYNAMIC_CHECK_INTERVAL_SECONDS,
        )
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(components.index)
    )
    return (
        east_displacement.to_numpy(dtype=float),
        north_displacement.to_numpy(dtype=float),
    )


def _relative_displacement_error(
    actual_east_displacement_m: NDArray[np.float64],
    actual_north_displacement_m: NDArray[np.float64],
    reported_east_displacement_m: NDArray[np.float64],
    reported_north_displacement_m: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return absolute vector error in meters and its fraction of actual motion."""
    actual_distance_m = np.hypot(
        actual_east_displacement_m, actual_north_displacement_m
    )
    error_distance_m = np.hypot(
        actual_east_displacement_m
        - reported_east_displacement_m,
        actual_north_displacement_m
        - reported_north_displacement_m,
    )
    relative_error = np.divide(
        error_distance_m,
        actual_distance_m,
        out=np.where(error_distance_m == 0, 0.0, np.inf),
        where=actual_distance_m != 0,
    )
    return error_distance_m, relative_error


def _position_change_violations(ordered: pd.DataFrame) -> pd.Series:
    """Return rows whose displacement error exceeds both configured limits."""
    grouped = ordered.groupby(FLIGHT_PATH_OUTPUT_COLUMN, sort=False)
    elapsed_seconds = (
        ordered["time"] - grouped["time"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS)
    ).dt.total_seconds()
    check_interval = elapsed_seconds.eq(DYNAMIC_CHECK_INTERVAL_SECONDS)

    actual_east, actual_north = _east_north_displacement(
        grouped["lat"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS).to_numpy(dtype=float),
        grouped["lon"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS).to_numpy(dtype=float),
        ordered["lat"].to_numpy(dtype=float),
        ordered["lon"].to_numpy(dtype=float),
    )
    reported_east, reported_north = _reported_east_north_displacement(
        ordered["velocity"].to_numpy(dtype=float),
        ordered["heading"].to_numpy(dtype=float),
        ordered[FLIGHT_PATH_OUTPUT_COLUMN].to_numpy(dtype=np.int64),
    )
    position_error_m, relative_error = _relative_displacement_error(
        actual_east,
        actual_north,
        reported_east,
        reported_north,
    )
    return check_interval & (
        (position_error_m > MAX_POSITION_ERROR_M)
        & (relative_error > MAX_DISPLACEMENT_ERROR_FRACTION)
    )


def _velocity_change_violations(ordered: pd.DataFrame) -> pd.Series:
    """Return rows whose average velocity change exceeds the configured limit."""
    grouped = ordered.groupby(FLIGHT_PATH_OUTPUT_COLUMN, sort=False)
    elapsed_seconds = (
        ordered["time"] - grouped["time"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS)
    ).dt.total_seconds()
    velocity_change = (
        grouped["velocity"].diff(DYNAMIC_CHECK_INTERVAL_SECONDS).abs()
        / elapsed_seconds
    )
    return elapsed_seconds.eq(DYNAMIC_CHECK_INTERVAL_SECONDS) & (
        velocity_change > MAX_VELOCITY_CHANGE_MPS_PER_SECOND
    )


def _altitude_change_violations(ordered: pd.DataFrame) -> pd.Series:
    """Return rows whose average altitude change exceeds the configured limit."""
    grouped = ordered.groupby(FLIGHT_PATH_OUTPUT_COLUMN, sort=False)
    elapsed_seconds = (
        ordered["time"] - grouped["time"].shift(DYNAMIC_CHECK_INTERVAL_SECONDS)
    ).dt.total_seconds()
    altitude_change = (
        grouped["geoaltitude"].diff(DYNAMIC_CHECK_INTERVAL_SECONDS).abs()
        / elapsed_seconds
    )
    return elapsed_seconds.eq(DYNAMIC_CHECK_INTERVAL_SECONDS) & (
        altitude_change > MAX_ALTITUDE_CHANGE_MPS
    )


def _clean_trajectories(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], FlightPathSplitReport]:
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
        "missing_values": 0,
        "out_of_range_values": 0,
        "position_change_exceeds_threshold": 0,
        "velocity_change_exceeds_threshold": 0,
        "altitude_change_exceeds_threshold": 0,
        "flight_path_too_short": 0,
    }

    cleaned, dropped["missing_values"] = _filter_missing_values(cleaned)
    cleaned, dropped["out_of_range_values"] = _filter_out_of_range_values(cleaned)

    # Sort and group the remaining paths for dynamic checks over the configured
    # comparison interval.
    ordered, start_split_count = _split_flight_paths(cleaned)
    starting_flight_paths = int(ordered[FLIGHT_PATH_OUTPUT_COLUMN].nunique())
    position_violation = _position_change_violations(ordered)
    velocity_violation = _velocity_change_violations(ordered)
    altitude_violation = _altitude_change_violations(ordered)

    # Drop only the row that violates a dynamic rule. Count overlapping
    # failures under the first rule so the reason totals remain exclusive.
    position_rows = position_violation.fillna(False)
    velocity_rows = velocity_violation.fillna(False) & ~position_rows
    altitude_rows = (
        altitude_violation.fillna(False) & ~position_rows & ~velocity_rows
    )
    dropped["position_change_exceeds_threshold"] = int(position_rows.sum())
    dropped["velocity_change_exceeds_threshold"] = int(velocity_rows.sum())
    dropped["altitude_change_exceeds_threshold"] = int(altitude_rows.sum())

    dynamic_row = position_rows | velocity_rows | altitude_rows
    cleaned = cleaned.loc[
        ~dynamic_row.reindex(cleaned.index, fill_value=False)
    ]

    # Re-segment after row-level filtering, then remove only short segments.
    cleaned, end_split_count = _split_flight_paths(cleaned)
    cleaned, dropped["flight_path_too_short"] = _filter_short_flight_paths(
        cleaned
    )

    return (
        cleaned.reset_index(drop=True),
        dropped,
        {
            "starting_flight_paths": starting_flight_paths,
            "remaining_flight_paths": int(
                cleaned[FLIGHT_PATH_OUTPUT_COLUMN].nunique()
            ),
            "flight_path_splits_at_start": start_split_count,
            "flight_path_splits_at_end": end_split_count,
        },
    )


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
    cleaned, dropped_rows, flight_path_report = _clean_trajectories(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return {
        "input_file": input_file,
        "input_rows": input_rows,
        "starting_flight_paths": flight_path_report["starting_flight_paths"],
        "output_rows": len(cleaned),
        "remaining_flight_paths": flight_path_report["remaining_flight_paths"],
        "flight_path_splits_at_start": flight_path_report[
            "flight_path_splits_at_start"
        ],
        "flight_path_splits_at_end": flight_path_report[
            "flight_path_splits_at_end"
        ],
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
    print(f"\nStarting flight paths: {report['starting_flight_paths']}.")
    print(
        "Flight path splits at start: "
        f"{report['flight_path_splits_at_start']}."
    )
    print(
        "Flight path splits at end: "
        f"{report['flight_path_splits_at_end']}."
    )
    print(
        f"\nRemaining flight paths of at least {MIN_FLIGHT_DURATION_MINUTES} minutes: "
        f"{report['remaining_flight_paths']}."
    )
    print("\nRows dropped (each row counted under its first failing reason):")
    for reason, count in report["dropped_rows"].items():
        print(f"  {reason.replace('_', ' ')}: {count}")


if __name__ == "__main__":
    main()
