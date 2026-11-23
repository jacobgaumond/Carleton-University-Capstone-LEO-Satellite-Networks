from base import *

import torch
from torch.utils.data import Dataset, DataLoader

from bisect import bisect_right

class AdsBDataSet(Dataset):
    """
    Considerations:
        - One should split their flight paths into training/validation/test lists *before* feeding them into an AdsBDataSet (i.e., use three
          separate AdsBDataSets).
    Attributes:
        self._flight_paths
            list[torch.Tensor] holding the flight paths of the DataSet.
        self._window_size
            int size of the sliding windows (i.e., fixed slice of consecutive data points) that are used as inputs during training.
            Note: We should start with window_size = 10, and experiment with 15 and 20 (maybe 30, if our machines can keep up).
        self._target_dims
            int number of Tensor dimensions (i.e., DataFrame/CSV feature columns) that are relevant for our target/desired output (e.g., lat,
            lon, and geoaltitude). These dimensions/columns should be stored at the front of the Tensor (i.e., the first 3 dimensions).
        self._cumulative_sliding_window_count
            list[int] where each entry with index i + 1 contains the total number of sliding windows in all paths up to self._flight_paths[i]
            (inclusive).
    """
    def __init__(self, flight_paths: list[torch.Tensor], window_size: int = 10):
        """
        Requirements:
            - The first 3 dimensions of flight_paths' Tensors (i.e., the first 3 columns of the DataFrame/CSV) must contain the flight path's
              lat, lon, and geoaltitude data (relates to how self._target_dims is used)
        """
        if window_size < 1:
            raise ValueError("AdsBDataSet.__init__: window_size must be at least 1")

        self._flight_paths = [path.clone() for path in flight_paths if len(path) > window_size] # Only use paths with at least 1 window slice.
                                                                                                # Cloning to avoid mutation via the list/Tensor reference.

        self._window_size = window_size
        self._target_dims = 3

        # _cumulative_sliding_window_count[i + 1] = total valid windows across groups 0..i
        # (i.e., _cumulative_sliding_window_count is a list, where each entry with index i + 1 contains the total number of sliding windows in all paths up to _flight_paths[i])
        self._cumulative_sliding_window_count = [0]
        for path in self._flight_paths:
            num_windows = max(0, path.shape[0] - window_size) # path.shape[0] is equivalent to len(path); it returns num of rows
            self._cumulative_sliding_window_count.append(self._cumulative_sliding_window_count[-1] + num_windows)

    def __len__(self) -> int:
        return self._cumulative_sliding_window_count[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Find path_idx such that _cumulative_sliding_window_count[path_idx] <= idx < _cumulative_sliding_window_count[path_idx + 1]
        # (i.e., find the path_idx identifying the path that contains the "idxth" sliding window)
        path_idx = bisect_right(self._cumulative_sliding_window_count, idx) - 1
        data_point_idx = idx - self._cumulative_sliding_window_count[path_idx] # A version of idx that points to the data point within the path (i.e., self._flight_paths[path_idx])

        # Get the item
        path = self._flight_paths[path_idx]

        ## x is the input window slice (i.e., historical data preceding our target)
        ## We are taking a slice of a path the size of self._window_size, starting from data_point_idx.
        x = path[data_point_idx : data_point_idx + self._window_size]       # Tensor shape: (window_size, num_features) or (W, F)

        ## y is the targetted output data (i.e., the desired data being predicted)
        ## We are indexing the data point after the window slice, and returning the first three values in that data point (i.e., lat, lon, geoaltitude).
        y = path[data_point_idx + self._window_size, : self._target_dims]   # Tensor shape: (target_dims,)
        return x, y
