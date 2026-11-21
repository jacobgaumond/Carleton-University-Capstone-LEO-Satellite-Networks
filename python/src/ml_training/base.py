import torch
from pathlib import Path
import pandas as pd

ROOT_DIR_PATH               = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_PARENT_PATH   = ROOT_DIR_PATH / "out" / "ml/data/processed"
DEFAULT_INPUT_FILENAME      = "adsb_trajectory_data_cleaned.csv"

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

def clean_ads_b_df(df: pd.DataFrame, cols_to_drop: list[str] = ["icao24"]) -> pd.DataFrame:
    """
    Cleans (prepares) an input DataFrame (of processed ADS-B data) for training, returning a new DataFrame object.

    Input DataFrames are expected to have no missing entries.
    """

    # df = df.copy() # Avoids parameter mutation; unnecessary since the next statement (df.drop) also copies

    # Drop columns cols_to_drop
    df = df.drop(columns=cols_to_drop)

    # Bring the "flight_path_id" column to the front
    col_list = df.columns.tolist()
    col_list.insert(0, col_list.pop(col_list.index("flight_path_id")))
    df = df[col_list]

    return df
