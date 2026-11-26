from base import *

import torch
from torch.utils.data import Dataset, DataLoader

from bisect import bisect_right

WINDOW_SIZE = 10 # We should start with window_size = 10, and experiment with 15 and 20 (maybe 30, if our machines can keep up).
HORIZON     = 5
TARGET_DIMS = 3

BATCH_SIZE  = 64 # We should experiment, starting with 64 or 128. Small batch sizes (like 16 or 32) produce noiser gradient estimates per step (can help regularization (i.e., avoiding overfitting)), while large batch sizes (128 or 256+) smooth out gradients at the cost of fewer weight updates per epoch (i.e., for the same size dataset) and memory/compute resources.
NUM_WORKERS = 0 # When NUM_WORKERS == 0, data loading is done synchronously in the main thread. When > 0, extra threads are spawned to do this; since we're using Tensors, the data is already in-memory, and using more threads will have little or no benefit, and can even harm due to the IPC overhead.

class AdsBDataset(Dataset):
    """
    Considerations:
        - One should split their flight paths into training/validation/test lists *before* feeding them into an AdsBDataset (i.e., use three
          separate AdsBDatasets).
    Attributes:
        self._flight_paths
            list[torch.Tensor] holding the flight paths of the Dataset.
        self._window_size
            int size of the sliding windows (i.e., fixed slice of consecutive data points) that are used as inputs during training.
        self._horizon
            int size of the horizons (i.e., fixed slice of consecutive data points that come directly after a sliding window) that are used
            as outputs during training.
        self._target_dims
            int number of Tensor dimensions (i.e., DataFrame/CSV feature columns) that are relevant for our target/desired output (e.g., lat,
            lon, and geoaltitude). These dimensions/columns should be stored at the front of the Tensor (i.e., the first 3 dimensions).
        self._cumulative_sliding_window_count
            list[int] where each entry with index i + 1 contains the total number of sliding windows in all paths up to self._flight_paths[i]
            (inclusive).
    """
    def __init__(self, flight_paths: list[torch.Tensor], window_size: int = WINDOW_SIZE, horizon: int = HORIZON):
        """
        Requirements:
            - The first 3 dimensions of flight_paths' Tensors (i.e., the first 3 columns of the DataFrame/CSV) must contain the flight path's
              lat, lon, and geoaltitude data (relates to how self._target_dims is used)
        """
        if window_size < 1:
            raise ValueError("AdsBDataset.__init__: window_size must be at least 1")
        if horizon < 1:
            raise ValueError("AdsBDataset.__init__: horizon must be at least 1")

        self._window_size = window_size
        self._horizon = horizon
        self._target_dims = TARGET_DIMS
        self._flight_paths = [path.clone() for path in flight_paths if          # Cloning to avoid mutation via the list/Tensor reference.
                              len(path) >= self._window_size + self._horizon]   # Only use paths with at least 1 sample (window slice & horizon).

        # _cumulative_sliding_window_count[i + 1] = total valid windows across groups 0..i
        # (i.e., _cumulative_sliding_window_count is a list, where each entry with index i + 1 contains the total number of sliding windows in all paths up to _flight_paths[i])
        self._cumulative_sliding_window_count = [0]
        for path in self._flight_paths:
            num_windows = max(0, path.shape[0] - self._window_size - self._horizon + 1) # path.shape[0] is equivalent to len(path); it returns num of rows
            self._cumulative_sliding_window_count.append(self._cumulative_sliding_window_count[-1] + num_windows)

    def __len__(self) -> int:
        return self._cumulative_sliding_window_count[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"AdsBDataset.__getitem__: Index {idx} out of range for dataset of length {len(self)}")

        # Find path_idx such that _cumulative_sliding_window_count[path_idx] <= idx < _cumulative_sliding_window_count[path_idx + 1]
        # (i.e., find the path_idx identifying the path that contains the "idxth" sliding window)
        path_idx = bisect_right(self._cumulative_sliding_window_count, idx) - 1
        data_point_idx = idx - self._cumulative_sliding_window_count[path_idx] # A version of idx that points to the data point within the path (i.e., self._flight_paths[path_idx])

        # Get the item
        path = self._flight_paths[path_idx]

        ## x is the input window slice (i.e., historical data preceding our target)
        ## We are taking a slice of a path the size of self._window_size, starting from data_point_idx.
        x = path[data_point_idx : data_point_idx + self._window_size]   # Tensor shape: (window_size, num_features) or (W, F)

        ## y is the targetted output data (i.e., the desired data being predicted)
        ## We are indexing a `horizon` amount of data points after the window slice, and returning the first three values in each data point (i.e., lat, lon, geoaltitude).
        y = path[data_point_idx + self._window_size : data_point_idx + self._window_size + self._horizon,
                 : self._target_dims]                                   # Tensor shape: (horizon, target_dims) or (H, D)
        return x, y

def createTrainValTestDataLoaders(flight_paths: list[torch.Tensor],
                                  train_val_test_split: tuple[int, int, int] = (70, 15, 15),
                                  batch_size: int = BATCH_SIZE,
                                  num_workers: int = NUM_WORKERS,
                                  ds_window_size: int = None, # Avoid changing; None defaults to AdsBDataset's default
                                  ) -> dict[str, AdsBDataset]:
    """
    Takes:
        - A list of Tensors containing ADS-B flight path data.
        - An optional specification for how the data should be split into training, validation, and testing data, stored in a tuple containing
          the percentages of each (in that order). The percentages must add to 100.
        - An optional window_size that overwrites the AdsBDataset default.

    Returns three DataLoaders in a dictionary, for training, validation, and testing. Each DataLoader is indexed with "train", "val", and
    "test", respectively.
    """
    SPLIT_TRAIN_INDEX = 0
    SPLIT_VAL_INDEX   = 1
    SPLIT_TEST_INDEX  = 2

    if sum(train_val_test_split) != 100:
        raise ValueError("createTrainValTestDataLoaders: train_val_test_split values must add to 100")
    num_train_paths = len(flight_paths) * train_val_test_split[SPLIT_TRAIN_INDEX] // 100
    num_val_paths   = len(flight_paths) * train_val_test_split[SPLIT_VAL_INDEX]   // 100

    datasets = {"train": None, "val": None, "test": None}
    if ds_window_size is None:
        datasets["train"]   = AdsBDataset(flight_paths[: num_train_paths])
        datasets["val"]     = AdsBDataset(flight_paths[num_train_paths : num_train_paths + num_val_paths])
        datasets["test"]    = AdsBDataset(flight_paths[num_train_paths + num_val_paths :])
    else:
        datasets["train"]   = AdsBDataset(flight_paths[: num_train_paths]                                   , ds_window_size)
        datasets["val"]     = AdsBDataset(flight_paths[num_train_paths : num_train_paths + num_val_paths]   , ds_window_size)
        datasets["test"]    = AdsBDataset(flight_paths[num_train_paths + num_val_paths :]                   , ds_window_size)

    for key, ds in datasets.items:
        if len(ds) < 1:
            raise ValueError(f"createTrainValTestDataLoaders: The {key} dataset has no sliding windows; increase the number of valid paths in flight_paths or adjust the percentages in train_val_test_split")

    dataLoaders = {
        "train":    DataLoader(datasets["train"],   batch_size=batch_size, num_workers=num_workers, shuffle=True),
        "val":      DataLoader(datasets["val"],     batch_size=batch_size, num_workers=num_workers, shuffle=False),
        "test":     DataLoader(datasets["test"],    batch_size=batch_size, num_workers=num_workers, shuffle=False),
                   }
    return dataLoaders
