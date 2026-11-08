# ADS-B history downloader

`python/src/adsb/download_adsb.py` downloads aircraft trajectory data from the OpenSky
historical database. The date range, geographic bounds, and output path are
read from a YAML configuration file.

## Setup

Install the project dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml` (the default config file path), then edit the values for the download you need. Note that `config/config.yaml` is ignored by Git so that local download settings do not get committed.

## Configuration

| Key | Description |
| --- | --- |
| `start_time` | Inclusive query start as a timezone-aware ISO 8601 timestamp. |
| `end_time` | Query end as a timezone-aware ISO 8601 timestamp, after `start_time`. |
| `min_lat`, `max_lat` | Latitude bounds, with `-90 <= min_lat < max_lat <= 90`. |
| `min_lon`, `max_lon` | Longitude bounds, with `-180 <= min_lon < max_lon <= 180`. |
| `output_file` | CSV destination; relative paths are resolved from the `out/` directory. |

Quote timestamps in YAML to keep them as strings, and include a timezone offset
such as `+00:00`. OpenSky request chunking uses PyOpenSky's default (one hour).

## Warnings

> Warning: PyOpenSky requires a valid OpenSky login and account access before
> historical queries will work. If you do not authenticate with the OpenSky
> account used for the dataset, requests may fail or return limited results.

> Warning: OpenSky applies strict Trino usage rules for historical data access.
> Queries must filter on the hourly/day partition column, large scans are rejected,
> users are limited to two concurrent queries and two queued queries, and queries
> are capped at 30 minutes. Repeatedly ignoring these limits can result in
> suspension, so keep each request narrow and break large ranges into smaller
> chunks.

## Run

From the repository root, using the default `adsb/config.yaml`:

```powershell
python adsb/src/download_adsb.py
```

To use a different config file:

```powershell
python adsb/src/download_adsb.py --config path/to/your-config.yaml
```

The output directory is created automatically when the query returns data.

## Filter by ICAO24

Extract rows for one aircraft from the history CSV:

```powershell
python adsb/src/filter_icao24.py --icao24 ABC123
```

The address must be six hexadecimal characters. By default, the script reads
`adsb/data/history/adsb_trajectory_data.csv` and writes matching rows to a
similarly named CSV beside it, such as
`adsb_trajectory_data_icao24_ABC123.csv`. You can specify different paths with
`--input` and `--output`:

```powershell
python adsb/src/filter_icao24.py --icao24 ABC123 --input path/to/input.csv --output path/to/filtered.csv
```

## Export MATLAB waypoints

Convert an extracted aircraft CSV into a MATLAB-ready waypoint CSV:

```powershell
python adsb/src/convert_adsb_waypoints.py `
  --input adsb/data/history/adsb_trajectory_data_icao24_A733DD.csv
```

By default, the output is written beside the input with `_waypoints` appended
to its name. Each row has a UTC timestamp, elapsed arrival time in seconds,
latitude and longitude in degrees, and geometric altitude in meters. Rows
missing coordinates or geometric altitude are skipped and counted in the
script's output. To choose another output path, add
`--output path/to/waypoints.csv`.

In MATLAB, load the waypoint CSV and create the trajectory:

```matlab
waypointData = readtable("adsb_trajectory_data_icao24_A733DD_waypoints.csv");
waypoints = [waypointData.latitude_deg, waypointData.longitude_deg, ...
    waypointData.altitude_m];
timeOfArrival = waypointData.time_of_arrival_s;
trajectory = geoTrajectory(waypoints, timeOfArrival);
```
