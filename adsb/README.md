# ADS-B history downloader

`src/download_adsb.py` downloads aircraft trajectory data from the OpenSky
historical database. The date range, geographic bounds, and output path are
read from a YAML configuration file.

## Setup

Install the project dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Copy `config.example.yaml` to `config.yaml`, then edit the values for the
download you need. The real `config.yaml` is ignored by Git so local download
settings do not get committed.

## Configuration

| Key | Description |
| --- | --- |
| `start_time` | Inclusive query start as a timezone-aware ISO 8601 timestamp. |
| `end_time` | Query end as a timezone-aware ISO 8601 timestamp, after `start_time`. |
| `min_lat`, `max_lat` | Latitude bounds, with `-90 <= min_lat < max_lat <= 90`. |
| `min_lon`, `max_lon` | Longitude bounds, with `-180 <= min_lon < max_lon <= 180`. |
| `output_file` | CSV destination; relative paths are resolved from the config file's directory. |

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
