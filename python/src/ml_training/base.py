import torch
from pathlib import Path
import pandas as pd

ROOT_DIR_PATH               = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_PARENT_PATH   = ROOT_DIR_PATH / "out" / "ml/data/processed"
DEFAULT_INPUT_FILENAME      = "adsb_trajectory_data_cleaned.csv"

# Note: If modifying NUMPY_FLOAT_DATA_TYPE, you should also modifying the Tensor `dtype` used the function `tensor_list_from_ads_b_df`
NUMPY_FLOAT_DATA_TYPE = "float32" # The lat, lon, geoaltitude, velocity, and heading are typecasted from float64, as its unsupported for MPS Tensors; we can experiment with non-MPS and float64 later, if needed.
FEATURE_COLS = ["time_diff", "lat", "lon", "geoaltitude", "velocity", "heading"]

# Setup hardware device

if torch.cuda.is_available():
    device = torch.device("cuda")   # For NVIDIA GPU's
elif torch.backends.mps.is_available():
    device = torch.device("mps")    # For Apple Silicon GPU's
else:
    device = torch.device("cpu")    # Fallback to CPU (not ideal)

torch.set_default_device(device)

# Basic function declarations

def df_from_csv(input_csv_filename: str = DEFAULT_INPUT_FILENAME, input_parent_path: Path = DEFAULT_INPUT_PARENT_PATH) -> pd.DataFrame:
    """
    Takes an input CSV file, and returns the data in a DataFrame.
    """

    return pd.read_csv(input_parent_path / input_csv_filename)

def clean_ads_b_df(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans (prepares) an input DataFrame (of processed ADS-B data) for training, returning a new DataFrame object.

    Input DataFrames have the following requirements:
        - No missing entries.
        - "time" entries must be in UTC (example: "2026-08-01 12:00:01+00:00")
        - Flight paths grouped by "flight_path_id" must:
            - Have all rows be contiguous.
            - Have all rows be sorted, such that the "time" column is strictly increasing.
    """
    cols_to_drop = ["icao24"] # Used to track cols to drop at the end
    output_df = input_df.copy() # Avoids parameter mutation

    # Change the "time" column (containing timestamps in the following form: "2026-08-01 12:00:01+00:00") into a
    # "time_diff" column (containing the time, in seconds, that has elapsed since the last data entry).
    output_df["time"] = pd.to_datetime(output_df["time"], utc=True)
    output_df["time_diff"] = (
        output_df.groupby("flight_path_id", sort=False)["time"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .astype(NUMPY_FLOAT_DATA_TYPE)
    )
    cols_to_drop += ["time"]

    # Drop columns
    output_df = output_df.drop(columns=cols_to_drop)

    # Reorder columns
    ## "flight_path_id" should be at the front
    ## "time_diff" should be next
    col_list = output_df.columns.tolist()
    col_list.insert(0, col_list.pop(col_list.index("flight_path_id")))
    col_list.insert(1, col_list.pop(col_list.index("time_diff")))
    ## The rest are done arbitrarily, but explicitly so that they are enforced
    col_list.insert(2, col_list.pop(col_list.index("lat")))
    col_list.insert(3, col_list.pop(col_list.index("lon")))
    col_list.insert(4, col_list.pop(col_list.index("geoaltitude")))
    col_list.insert(5, col_list.pop(col_list.index("velocity")))
    col_list.insert(6, col_list.pop(col_list.index("heading")))

    output_df = output_df[col_list]

    # Set all non-ID data (i.e., data for feature columns) to the same float type (for training)
    # output_df["time_diff"] = output_df["time_diff"].astype(NUMPY_FLOAT_DATA_TYPE) # Already done (above)
    output_df["lat"] = output_df["lat"].astype(NUMPY_FLOAT_DATA_TYPE)
    output_df["lon"] = output_df["lon"].astype(NUMPY_FLOAT_DATA_TYPE)
    output_df["geoaltitude"] = output_df["geoaltitude"].astype(NUMPY_FLOAT_DATA_TYPE)
    output_df["velocity"] = output_df["velocity"].astype(NUMPY_FLOAT_DATA_TYPE)
    output_df["heading"] = output_df["heading"].astype(NUMPY_FLOAT_DATA_TYPE)

    return output_df

def tensor_list_from_ads_b_df(df: pd.DataFrame) -> list[torch.tensor]:
    """
    Takes an input DataFrame of processed & cleaned ADS-B data, and unpackages it into a list of tensors (where
    each tensor represents data from a distinct flight path) for ML training.
    """

    tensor_list = [
        torch.tensor(flight_path_df[FEATURE_COLS].to_numpy(dtype=NUMPY_FLOAT_DATA_TYPE), dtype=torch.float32) \
            for _, flight_path_df in df.groupby("flight_path_id", sort=False)
    ]

    return tensor_list
