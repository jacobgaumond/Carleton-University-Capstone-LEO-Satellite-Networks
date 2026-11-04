# ADS-B parser for the MATLAB sim (WIP, separate from the ML team's)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = "adsb/data/samples"
DEFAULT_OUTPUT = "adsb/data/parsed_flights"
MAX_SPEED_MPS = 350
MAX_GAP_SECONDS = 30
EARTH_RADIUS_M = 6_371_000
OUTPUT_COLUMNS = ["time", "icao24", "lat", "lon", "geoaltitude", "velocity", "heading"]


def _distance_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _drop_jumps(flight: pd.DataFrame) -> pd.DataFrame:
    # keep a point only if it's reachable from the last kept one
    lat, lon, t = flight["lat"].to_numpy(), flight["lon"].to_numpy(), flight["pos_time"].to_numpy()
    keep = np.zeros(len(flight), dtype=bool)
    keep[0] = True
    last = 0
    for i in range(1, len(flight)):
        speed = _distance_m(lat[last], lon[last], lat[i], lon[i]) / (t[i] - t[last])
        if speed <= MAX_SPEED_MPS:
            keep[i] = True
            last = i
    return flight.loc[keep]


def parse_flight(flight: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"input rows": len(flight)}

    usable = (flight["time"].notna() & flight["lat"].between(-90, 90)
              & flight["lon"].between(-180, 180) & flight["geoaltitude"].notna()
              & flight["lastposupdate"].notna())
    report["missing values"] = int((~usable).sum())
    flight = flight.loc[usable]

    # opensky repeats old positions, so use the real position time and drop repeats
    flight = flight.assign(pos_time=flight["lastposupdate"]).sort_values("pos_time")
    new_position = flight["pos_time"].diff().fillna(1) > 0
    report["stale positions"] = int((~new_position).sum())
    flight = flight.loc[new_position]

    before = len(flight)
    if before:
        flight = _drop_jumps(flight)
    report["impossible jumps"] = before - len(flight)
    report["output rows"] = len(flight)

    if len(flight) < 2:
        report.update(max_gap_s=None, duration_min=0, status="rejected: too few points")
        return flight, report

    max_gap = float(flight["pos_time"].diff().max())
    report["max_gap_s"] = round(max_gap, 1)
    report["duration_min"] = round((flight["pos_time"].iloc[-1] - flight["pos_time"].iloc[0]) / 60)
    report["status"] = "usable" if max_gap < MAX_GAP_SECONDS else f"rejected: gap >= {MAX_GAP_SECONDS} s"

    flight = flight.assign(time=pd.to_datetime(flight["pos_time"], unit="s", utc=True)
                           .dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3] + "+00:00")
    return flight[OUTPUT_COLUMNS], report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean OpenSky ADS-B CSVs into one file per usable flight.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV file or folder of CSVs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="folder for cleaned flights")
    args = parser.parse_args()

    source = Path(args.input)
    files = [source] if source.is_file() else sorted(p for p in source.glob("*.csv") if p.name != "manifest.csv")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in files:
        # icao24 as text so leading zeros survive
        data = pd.read_csv(path, dtype={"icao24": "string", "callsign": "string"})
        data["icao24"] = data["icao24"].str.strip().str.lower()
        data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
        data = data[data["icao24"].notna() & data["icao24"].ne("")]
        aircraft = data["icao24"].unique()
        for icao24 in aircraft:
            cleaned, report = parse_flight(data[data["icao24"] == icao24])
            name = path.stem if len(aircraft) == 1 else f"{path.stem}_{icao24}"
            if report["status"] == "usable":
                cleaned.to_csv(out_dir / f"{name}.csv", index=False)
            reports.append({"flight": name, "icao24": icao24, **report})

    summary = pd.DataFrame(reports).set_index("flight")
    summary.to_csv(out_dir / "report.csv")
    print(summary.to_string())
    usable = (summary["status"] == "usable").sum()
    print(f"\n{usable} of {len(summary)} flights usable; written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
