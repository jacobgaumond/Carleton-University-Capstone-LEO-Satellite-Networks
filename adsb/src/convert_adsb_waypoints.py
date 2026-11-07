"""Convert an extracted ADS-B CSV into MATLAB geoTrajectory waypoints."""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path


WAYPOINT_FIELDS = (
    "timestamp_utc",
    "time_of_arrival_s",
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
)
REQUIRED_INPUT_FIELDS = {"time", "lat", "lon", "geoaltitude"}


def _parse_number(value: str | None, field: str, row_number: int) -> float | None:
    if value is None or not value.strip():
        return None

    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"CSV row {row_number}: '{field}' must be a number."
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"CSV row {row_number}: '{field}' must be finite.")
    return number


def _parse_timestamp(value: str | None, row_number: int) -> datetime:
    if value is None or not value.strip():
        raise ValueError(f"CSV row {row_number}: 'time' must not be empty.")

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"CSV row {row_number}: 'time' must be a valid ISO 8601 timestamp."
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"CSV row {row_number}: 'time' must include a timezone.")
    return timestamp.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an extracted ADS-B CSV into MATLAB geoTrajectory waypoints."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Extracted ADS-B CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output waypoint CSV (default: next to input with '_waypoints' suffix).",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_waypoints.csv")
    )
    if input_path == output_path:
        parser.error("Output path must not be the same as the input path.")

    rows: list[tuple[datetime, float, float, float]] = []
    skipped_rows = 0
    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            parser.error("Input CSV must have a header row.")
        missing_fields = REQUIRED_INPUT_FIELDS - set(reader.fieldnames)
        if missing_fields:
            parser.error(
                "Input CSV is missing required columns: "
                + ", ".join(sorted(missing_fields))
            )

        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_timestamp(row.get("time"), row_number)
            latitude = _parse_number(row.get("lat"), "lat", row_number)
            longitude = _parse_number(row.get("lon"), "lon", row_number)
            altitude = _parse_number(row.get("geoaltitude"), "geoaltitude", row_number)

            if latitude is None or longitude is None or altitude is None:
                skipped_rows += 1
                continue
            if not -90 <= latitude <= 90:
                raise ValueError(
                    f"CSV row {row_number}: 'lat' must be between -90 and 90."
                )
            if not -180 <= longitude <= 180:
                raise ValueError(
                    f"CSV row {row_number}: 'lon' must be between -180 and 180."
                )
            rows.append((timestamp, latitude, longitude, altitude))

    if not rows:
        parser.error("Input CSV has no rows with valid coordinates and geoaltitude.")

    rows.sort(key=lambda row: row[0])
    if any(current[0] <= previous[0] for previous, current in zip(rows, rows[1:])):
        parser.error("Valid waypoint timestamps must be unique.")

    start_time = rows[0][0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=WAYPOINT_FIELDS)
        writer.writeheader()
        for timestamp, latitude, longitude, altitude in rows:
            writer.writerow(
                {
                    "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "time_of_arrival_s": (timestamp - start_time).total_seconds(),
                    "latitude_deg": latitude,
                    "longitude_deg": longitude,
                    "altitude_m": altitude,
                }
            )

    print(f"Saved {len(rows)} waypoints to {output_path}")
    if skipped_rows:
        print(
            f"Skipped {skipped_rows} rows with missing coordinates or geoaltitude."
        )


if __name__ == "__main__":
    main()
