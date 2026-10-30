"""Download ADS-B trajectory data from the OpenSky historical database."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import yaml
from pyopensky.trino import Trino


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
REQUIRED_CONFIG_KEYS = {
    "start_time",
    "end_time",
    "min_lat",
    "max_lat",
    "min_lon",
    "max_lon",
    "output_file",
}


@dataclass(frozen=True)
class DownloadConfig:
    start_time: datetime
    end_time: datetime
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    output_file: Path


def _parse_timestamp(value: Any, key: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Config '{key}' must be an ISO 8601 timestamp string.")

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"Config '{key}' must be a valid ISO 8601 timestamp."
        ) from error

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"Config '{key}' must include a timezone offset.")
    return timestamp


def _parse_coordinate(config: dict[str, Any], key: str) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config '{key}' must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"Config '{key}' must be finite.")
    return float(value)


def _load_config(config_path: Path) -> DownloadConfig:
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError("The YAML config must contain a mapping of input values.")
    if not all(isinstance(key, str) for key in raw_config):
        raise ValueError("All YAML config keys must be strings.")

    missing_keys = REQUIRED_CONFIG_KEYS - raw_config.keys()
    extra_keys = raw_config.keys() - REQUIRED_CONFIG_KEYS
    if missing_keys:
        raise ValueError(f"Missing config keys: {', '.join(sorted(missing_keys))}")
    if extra_keys:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(extra_keys))}")

    start_time = _parse_timestamp(raw_config["start_time"], "start_time")
    end_time = _parse_timestamp(raw_config["end_time"], "end_time")
    if end_time <= start_time:
        raise ValueError("Config 'end_time' must be after 'start_time'.")

    min_lat = _parse_coordinate(raw_config, "min_lat")
    max_lat = _parse_coordinate(raw_config, "max_lat")
    min_lon = _parse_coordinate(raw_config, "min_lon")
    max_lon = _parse_coordinate(raw_config, "max_lon")
    if not -90 <= min_lat < max_lat <= 90:
        raise ValueError("Latitude bounds must satisfy -90 <= min_lat < max_lat <= 90.")
    if not -180 <= min_lon < max_lon <= 180:
        raise ValueError(
            "Longitude bounds must satisfy -180 <= min_lon < max_lon <= 180."
        )

    output_file_value = raw_config["output_file"]
    if not isinstance(output_file_value, str) or not output_file_value.strip():
        raise ValueError("Config 'output_file' must be a non-empty path string.")
    output_file = Path(output_file_value)
    if not output_file.is_absolute():
        output_file = config_path.resolve().parent / output_file

    return DownloadConfig(
        start_time=start_time,
        end_time=end_time,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        output_file=output_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ADS-B history using the values in a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG}).",
    )
    args = parser.parse_args()
    config = _load_config(args.config)

    bounds = (
        config.min_lon,
        config.min_lat,
        config.max_lon,
        config.max_lat,
    )

    print("ADS-B query summary:")
    print(f"  Start time: {config.start_time.isoformat()}")
    print(f"  End time:   {config.end_time.isoformat()}")
    print(
        "  Bounds: "
        f"lat={config.min_lat}..{config.max_lat}, "
        f"lon={config.min_lon}..{config.max_lon}"
    )
    print(f"  Output file: {config.output_file}")

    trajectory_data = Trino().history(
        start=config.start_time,
        stop=config.end_time,
        bounds=bounds,
    )

    if trajectory_data is not None and not trajectory_data.empty:
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        trajectory_data.to_csv(config.output_file, index=False)
        print(
            f"Success! Saved {len(trajectory_data)} rows to {config.output_file}"
        )
    else:
        print("No trajectory data found for this timeframe/bounding box.")


if __name__ == "__main__":
    main()
