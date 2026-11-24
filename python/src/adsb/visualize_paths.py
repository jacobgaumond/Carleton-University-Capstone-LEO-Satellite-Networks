"""Plot ADS-B trajectories in 3D for one or more ICAO24 addresses."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.axes3d import Axes3D

PLOT_COLUMNS = ["time", "icao24", "lat", "lon", "geoaltitude"]
DEFAULT_INPUT = "adsb/data/history/adsb_trajectory_data.csv"
PATH_SEGMENT_COLUMN = "_path_segment"


def _read_icao_path(
    input_path: Path,
    icao_address: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Read only the requested aircraft tracks and assign path segments."""
    data = pd.read_csv(
        input_path,
        usecols=PLOT_COLUMNS,
        dtype={"icao24": "string"},
    )
    data["icao24"] = data["icao24"].str.strip().str.lower()
    icao_address = icao_address.strip().lower()
    data = data.loc[data["icao24"] == icao_address].copy()

    if data.empty:
        raise ValueError("No rows found for ICAO address: " + icao_address)

    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    for column in ("lat", "lon", "geoaltitude"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["geoaltitude"] = data["geoaltitude"].fillna(-1)

    start = pd.to_datetime(start_time, utc=True) if start_time else None
    end = pd.to_datetime(end_time, utc=True) if end_time else None
    if start is not None and end is not None and start > end:
        raise ValueError(
            "The start time must be earlier than or equal to the end time."
        )
    if start is not None:
        data = data.loc[data["time"].ge(start)]
    if end is not None:
        data = data.loc[data["time"].le(end)]

    coordinates = data[["lat", "lon", "geoaltitude"]].to_numpy(
        dtype=float, na_value=float("nan")
    )
    finite_coordinates = pd.Series(
        [
            all(math.isfinite(value) for value in coordinate)
            for coordinate in coordinates
        ],
        index=data.index,
        dtype="bool",
    )
    data = data.loc[data["time"].notna() & finite_coordinates].copy()

    if data.empty:
        raise ValueError(
            "The requested ICAO address(es) have no plottable time and "
            "position data."
        )

    data = data.sort_values("time")
    time_diffs = data.groupby("icao24", sort=False)["time"].diff()
    starts_segment = time_diffs.isna() | time_diffs.ge(pd.Timedelta(seconds=10))
    return data.assign(**{PATH_SEGMENT_COLUMN: starts_segment.cumsum()})


def _plot_icao_path(
    axis: Axes3D,
    path: pd.DataFrame,
    icao_address: str,
    title_label: str | None = None,
) -> None:
    """Plot all time-contiguous segments for one aircraft."""
    for segment_number, (_, segment) in enumerate(
        path.groupby(PATH_SEGMENT_COLUMN, sort=False),
        start=1,
    ):
        segment = segment.sort_values("time")
        axis.plot(
            segment["lon"],
            segment["lat"],
            segment["geoaltitude"],
            marker="o",
            markersize=2,
            linewidth=1,
            label=f"Segment {segment_number}",
        )

    title = f"3D flight path: {icao_address.upper()}"
    if title_label:
        title += f" - {title_label}"
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_zlabel("Altitude (m)")
    axis.grid(True, alpha=0.3)
    if path[PATH_SEGMENT_COLUMN].nunique() > 1:
        axis.legend()


def visualize_paths(
    input_location: str,
    icao_address: str,
    start_time: str | None = None,
    end_time: str | None = None,
    title_label: str | None = None,
) -> None:
    """Show one 3D plot per requested ICAO24 address."""
    input_path = Path(input_location).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    normalized_address = icao_address.strip().lower()
    if not normalized_address:
        raise ValueError("Provide a non-empty ICAO24 address.")

    data = _read_icao_path(
        input_path,
        normalized_address,
        start_time=start_time,
        end_time=end_time,
    )
    path = data.loc[data["icao24"].eq(normalized_address)]
    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")
    _plot_icao_path(axis, path, normalized_address, title_label)
    figure.tight_layout()

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 3D ADS-B paths for one or more ICAO24 addresses."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--icao",
        required=True,
        help="One or more ICAO24 addresses to plot.",
    )
    parser.add_argument(
        "--start",
        help="Optional inclusive UTC start time (ISO 8601).",
    )
    parser.add_argument(
        "--end",
        help="Optional inclusive UTC end time (ISO 8601).",
    )
    parser.add_argument(
        "--label",
        help="Optional text appended to each plot title, such as a filter name.",
    )

    args = parser.parse_args()
    visualize_paths(
        args.input,
        args.icao,
        start_time=args.start,
        end_time=args.end,
        title_label=args.label,
    )


if __name__ == "__main__":
    main()
