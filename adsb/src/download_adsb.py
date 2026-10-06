"""
This is a POC script to download ADS-B trajectory data from the OpenSky Network historical database using the pyopensky library.
It will be adjusted once data constraints are decided.

For now, adjust START_TIME, END_TIME, and the bounding box (MIN_LAT, MAX_LAT, MIN_LON, MAX_LON) to your desired values.
"""

from pyopensky.trino import Trino
from datetime import datetime, timezone, timedelta
import os

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_TIME = datetime(
    2026, 8, 1, 12, 0, 0,
    tzinfo=timezone.utc
)

END_TIME = datetime(
    2026, 8, 1, 12, 30, 0,
    tzinfo=timezone.utc
)

# Example: Eastern Canada / northeastern US
MIN_LAT = 40.0
MAX_LAT = 50.0
MIN_LON = -85.0
MAX_LON = -60.0

bounds_tuple = (MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)

OUTPUT_FILE = "data/history/adsb_trajectory_data.csv"


# ---------------------------------------------------------
# Connect to OpenSky historical database
# ---------------------------------------------------------

trino = Trino()

trajectory_data = trino.history(
    start=START_TIME, 
    stop=END_TIME, 
    bounds=bounds_tuple,
    date_delta=timedelta(minutes=5) 
)

# ---------------------------------------------------------
# Store the trajectory data in a CSV
# ---------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

if trajectory_data is not None and not trajectory_data.empty:
    trajectory_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Saved {len(trajectory_data)} rows to {OUTPUT_FILE}")
else:
    print("No trajectory data found for this timeframe/bounding box.")
