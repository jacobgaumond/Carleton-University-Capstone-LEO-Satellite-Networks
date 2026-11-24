"""Create a stricter, simpler cleaned ADS-B trajectory CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT_DIR / "out" / "adsb" / "data" / "history" / "adsb_trajectory_data.csv"
)
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "out"
    / "ml"
    / "data"
    / "processed"
    / "adsb_trajectory_data_cleaned_strict_v2.csv"
)

MAX_ALTITUDE_M = 30_000
MAX_VELOCITY_MPS = 400
MAX_POSITION_ERROR_M = 50
MAX_DISPLACEMENT_ERROR_FRACTION = 0.25
MAX_VELOCITY_CHANGE_MPS_PER_SECOND = 10
MAX_ALTITUDE_CHANGE_MPS = 30
MIN_FLIGHT_GAP_SECONDS = 2
MIN_FLIGHT_DURATION_MINUTES = 6
EARTH_RADIUS_M = 6_371_000

LOCATION_COLUMNS = ["lat", "lon", "geoaltitude"]
OUTPUT_COLUMNS = [
    "time",
    "icao24",
    "lat",
    "lon",
    "geoaltitude",
    "velocity",
    "heading",
]
PATH_COLUMN = "flight_path_id"


def _assign_flight_paths(data: pd.DataFrame) -> pd.DataFrame:
    """Sort observations and start a path after a long gap or aircraft change."""
    data = data.sort_values(["icao24", "time"], kind="stable").copy()
    gap = data.groupby("icao24", sort=False)["time"].diff()
    new_path = gap.isna() | gap.ge(pd.Timedelta(seconds=MIN_FLIGHT_GAP_SECONDS))
    data[PATH_COLUMN] = new_path.cumsum()
    return data


def _remove_repeated_locations(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove a row when its lat/lon/altitude matches the previous row in its path."""
    previous_location = data.groupby(PATH_COLUMN, sort=False)[LOCATION_COLUMNS].shift()
    repeated = data[LOCATION_COLUMNS].eq(previous_location).all(axis=1)
    return data.loc[~repeated].copy(), int(repeated.sum())


def _east_north_displacement(
    lat_1: np.ndarray,
    lon_1: np.ndarray,
    lat_2: np.ndarray,
    lon_2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one pair of latitude/longitude coordinates to local meters."""
    lat_1 = np.deg2rad(lat_1)
    lat_2 = np.deg2rad(lat_2)
    lon_1 = np.deg2rad(lon_1)
    lon_2 = np.deg2rad(lon_2)
    delta_lon = (lon_2 - lon_1 + np.pi) % (2 * np.pi) - np.pi
    mean_lat = (lat_1 + lat_2) / 2
    east = EARTH_RADIUS_M * np.cos(mean_lat) * delta_lon
    north = EARTH_RADIUS_M * (lat_2 - lat_1)
    return east, north


def _find_bad_points(data: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Check each observation against only its immediately previous observation."""
    grouped = data.groupby(PATH_COLUMN, sort=False)
    elapsed = (data["time"] - grouped["time"].shift()).dt.total_seconds()
    valid_interval = elapsed.gt(0)

    previous_lat = grouped["lat"].shift().to_numpy(dtype=float)
    previous_lon = grouped["lon"].shift().to_numpy(dtype=float)
    actual_east, actual_north = _east_north_displacement(
        previous_lat,
        previous_lon,
        data["lat"].to_numpy(dtype=float),
        data["lon"].to_numpy(dtype=float),
    )

    # Estimate the movement between these two observations using the previous
    # row's reported speed and heading, multiplied by the actual elapsed time.
    previous_speed = grouped["velocity"].shift().to_numpy(dtype=float)
    previous_heading = np.deg2rad(grouped["heading"].shift().to_numpy(dtype=float))
    elapsed_values = elapsed.to_numpy(dtype=float)
    expected_east = previous_speed * np.sin(previous_heading) * elapsed_values
    expected_north = previous_speed * np.cos(previous_heading) * elapsed_values

    position_error = np.hypot(
        actual_east - expected_east,
        actual_north - expected_north,
    )
    actual_distance = np.hypot(actual_east, actual_north)
    relative_error = np.divide(
        position_error,
        actual_distance,
        out=np.where(position_error == 0, 0.0, np.inf),
        where=actual_distance != 0,
    )
    bad_position = valid_interval & (
        (position_error > MAX_POSITION_ERROR_M)
        & (relative_error > MAX_DISPLACEMENT_ERROR_FRACTION)
    )

    # Sudden speed or altitude changes are checked over this same adjacent pair.
    speed_change_per_second = grouped["velocity"].diff().abs() / elapsed
    bad_speed = valid_interval & (
        speed_change_per_second > MAX_VELOCITY_CHANGE_MPS_PER_SECOND
    )
    altitude_change_per_second = grouped["geoaltitude"].diff().abs() / elapsed
    bad_altitude = valid_interval & (
        altitude_change_per_second > MAX_ALTITUDE_CHANGE_MPS
    )

    reasons = {
        "position inconsistency": int(bad_position.sum()),
        "speed change": int(bad_speed.sum()),
        "altitude change": int(bad_altitude.sum()),
    }
    bad_point = bad_position | bad_speed | bad_altitude
    return bad_point, reasons


def process_data(input_path: Path, output_path: Path) -> None:
    """Clean one history CSV and write a separate strict output CSV."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if input_path == output_path:
        raise ValueError("The output must not overwrite the input history CSV.")

    data = pd.read_csv(input_path, usecols=OUTPUT_COLUMNS)
    input_rows = len(data)

    # Parse values first so missing and invalid measurements can be rejected.
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    data["icao24"] = data["icao24"].astype("string").str.strip()
    for column in ["lat", "lon", "geoaltitude", "velocity", "heading"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing = data[OUTPUT_COLUMNS].isna().any(axis=1) | data["icao24"].eq("")
    data = data.loc[~missing].copy()
    if data.empty:
        raise ValueError("No rows with complete, validly parsed values were found.")

    out_of_range = (
        ~data["lat"].between(-90, 90)
        | ~data["lon"].between(-180, 180)
        | ~data["geoaltitude"].between(0, MAX_ALTITUDE_M)
        | ~data["velocity"].between(0, MAX_VELOCITY_MPS)
        | ~data["heading"].between(0, 360)
    )
    data = data.loc[~out_of_range].copy()

    # Assign paths before duplicate removal, so identical positions from
    # different flights are never treated as duplicates.
    data = _assign_flight_paths(data)
    data, duplicate_count = _remove_repeated_locations(data)

    # Remove the current point whenever any adjacent-pair dynamic check fails.
    bad_points, bad_reasons = _find_bad_points(data)
    dynamic_count = int(bad_points.sum())
    data = data.loc[~bad_points].copy()

    # Rebuild paths after removals, then discard paths shorter than six minutes.
    data = _assign_flight_paths(data)
    if not data.empty:
        durations = data.groupby(PATH_COLUMN, sort=False)["time"].agg(
            lambda times: times.max() - times.min()
        )
        short_paths = durations.index[
            durations < pd.Timedelta(minutes=MIN_FLIGHT_DURATION_MINUTES)
        ]
        short_rows = data[PATH_COLUMN].isin(short_paths)
        short_count = int(short_rows.sum())
        data = data.loc[~short_rows].copy()
    else:
        short_count = 0

    output_columns = OUTPUT_COLUMNS + [PATH_COLUMN]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data[output_columns].to_csv(output_path, index=False)

    print(f"Read {input_rows} rows.")
    print(f"Removed incomplete rows: {int(missing.sum())}")
    print(f"Removed out-of-range rows: {int(out_of_range.sum())}")
    print(f"Removed back-to-back duplicate locations: {duplicate_count}")
    print(f"Removed dynamic outlier rows: {dynamic_count}")
    for reason, count in bad_reasons.items():
        print(f"  {reason}: {count}")
    print(f"Removed rows in paths shorter than six minutes: {short_count}")
    print(f"Wrote {len(data)} rows to {output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    process_data(args.input, args.output)


if __name__ == "__main__":
    main()
