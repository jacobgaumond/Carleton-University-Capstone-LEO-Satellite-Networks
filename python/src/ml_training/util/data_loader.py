"""Load trajectory CSV data and prepare model-ready datasets and DataLoaders.

Input data is grouped by ``flight_path_id`` and split by whole flight paths.
Windows are built only from positions sampled exactly one second apart, so a
history or forecast never crosses a flight boundary or a timestamp gap.
"""

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset


class PositionWindows(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Make feature-history to position-forecast examples from continuous runs.

    Flight paths should be split into training and validation sets before
    constructing datasets, so windows from one flight cannot occur in both.
    Each input contains ``history_seconds`` rows of features, and each target
    contains the following ``forecast_seconds`` rows of output positions.

    Attributes:
        segments: Dictionaries containing scaled feature and position runs
            plus raw position targets, kept separate by path and time gaps.
        cumulative_window_counts: Cumulative number of valid windows through
            each segment, used to map a dataset index to its segment.
        history_seconds: Number of rows returned in each input window.
        forecast_seconds: Number of future observations returned in each target.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_scaler: StandardScaler,
        position_scaler: StandardScaler,
        feature_columns: list[str],
        position_columns: list[str],
        history_seconds: int,
        forecast_seconds: int,
    ) -> None:
        """Create windows from the supplied path data.

        Args:
            data: Timestamp-ordered observations with ``time``,
                ``flight_path_id``, and all requested feature and output columns.
            feature_scaler: Fitted scaler for model input features.
            position_scaler: Fitted scaler for predicted output positions.
            feature_columns: Ordered columns supplied to the model as input.
            position_columns: Ordered columns predicted by the model.
            history_seconds: Number of one-second feature rows in each input.
            forecast_seconds: Number of one-second output rows in each target.

        Raises:
            ValueError: If no uninterrupted run can supply a full window.
        """
        # Each segment is a continuous run; a flight path or timestamp gap
        # always starts a separate run.
        self.segments: list[dict[str, np.ndarray]] = []
        # Cumulative counts allow global dataset indices to be mapped to runs.
        self.cumulative_window_counts: list[int] = []
        window_count = 0

        for _, path in data.groupby("flight_path_id", sort=False):
            elapsed_seconds = path["time"].diff().dt.total_seconds().to_numpy()
            features = path[feature_columns].to_numpy(dtype=np.float32)
            positions = path[position_columns].to_numpy(dtype=np.float32)
            # The first diff is NaN, which also starts the first run; all later
            # non-one-second intervals mark a gap between continuous samples.
            breaks = np.flatnonzero(elapsed_seconds[1:] != 1.0) + 1

            for feature_run, position_run in zip(
                np.split(features, breaks),
                np.split(positions, breaks),
            ):
                # The +1 counts the window whose last target is the run's last row.
                count = len(feature_run) - history_seconds - forecast_seconds + 1
                if count <= 0:
                    continue
                # Apply training-only scalers consistently to both data splits.
                scaled_features = feature_scaler.transform(feature_run).astype(
                    np.float32
                )
                scaled_positions = position_scaler.transform(position_run).astype(
                    np.float32
                )
                # Keep raw targets as well: distance metrics need geographic
                # coordinates, while model loss uses standardized positions.
                self.segments.append(
                    {
                        "scaled_features": scaled_features,
                        "scaled_positions": scaled_positions,
                        "raw_positions": position_run,
                    }
                )
                window_count += count
                self.cumulative_window_counts.append(window_count)

        if not self.cumulative_window_counts:
            raise ValueError(
                "No uninterrupted one-second path is long enough for "
                f"{history_seconds} seconds of history and "
                f"{forecast_seconds} seconds of future positions."
            )

        self.history_seconds = history_seconds
        self.forecast_seconds = forecast_seconds

    def __len__(self) -> int:
        """Return the number of valid history-to-forecast windows."""
        return self.cumulative_window_counts[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one scaled feature history and its following position target.

        Args:
            index: Global window index in this dataset.

        Returns:
            A pair of tensors shaped ``[history_seconds, input_features]`` and
            ``[forecast_seconds, output_positions]``.
        """
        # Find which continuous run contains this sliding-window index.
        segment_index = bisect_right(self.cumulative_window_counts, index)
        # Convert the global index to an offset local to that run.
        previous_count = (
            self.cumulative_window_counts[segment_index - 1] if segment_index > 0 else 0
        )
        start = index - previous_count
        segment = self.segments[segment_index]
        scaled_features = segment["scaled_features"]
        scaled_positions = segment["scaled_positions"]
        # Return observed features and the following ground-truth positions.
        history = scaled_features[start : start + self.history_seconds]
        future_start = start + self.history_seconds
        future = scaled_positions[future_start : future_start + self.forecast_seconds]
        return torch.from_numpy(history), torch.from_numpy(future)

    def iter_validation_batches(
        self,
        batch_size: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield model-ready histories and raw targets for non-overlapping forecasts.

        Training windows overlap at every possible start. Validation metric
        windows instead start one forecast horizon apart so their targets do
        not overlap.

        Args:
            batch_size: Maximum number of forecast windows per yielded batch.

        Yields:
            Standardized feature histories shaped
            ``[batch, history, input_features]`` and matching unscaled position
            targets shaped ``[batch, forecast, output_positions]``.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("Validation batch size must be positive.")

        # Each forecast sample needs a complete input history and target.
        required_rows = self.history_seconds + self.forecast_seconds
        for segment in self.segments:
            scaled_features = segment["scaled_features"]
            raw_positions = segment["raw_positions"]
            # This is inclusive: its history and forecast end exactly at the
            # run's final observation.
            last_start = len(raw_positions) - required_rows
            # Advancing by one forecast horizon keeps target intervals disjoint.
            starts = range(0, last_start + 1, self.forecast_seconds)
            # Split the available forecast starts into memory-bounded batches.
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                # Inputs are already standardized for the model.
                histories = np.stack(
                    [
                        scaled_features[start : start + self.history_seconds]
                        for start in batch_starts
                    ]
                )
                # Keep actual targets unscaled so displacement metrics use
                # latitude/longitude degrees and altitude meters.
                actual_positions = np.stack(
                    [
                        raw_positions[
                            start
                            + self.history_seconds : start
                            + self.history_seconds
                            + self.forecast_seconds
                        ]
                        for start in batch_starts
                    ]
                )
                yield histories, actual_positions


@dataclass(frozen=True)
class TrainingData:
    """Prepared data and loaders shared by training and validation.

    The dataframes contain path-disjoint splits. Separate scalers are fitted
    to training input features and output positions, then shared with validation.

    Attributes:
        training_data: Rows belonging to training flight paths.
        validation_data: Rows belonging to held-out validation flight paths.
        heldout_data: Optional rows reserved for final testing.
        feature_scaler: Scaler fitted on input features in ``training_data``.
        position_scaler: Scaler fitted on output positions in ``training_data``.
        training_set: Window dataset created from the training paths.
        validation_set: Window dataset created from the validation paths.
        heldout_set: Optional windows dataset created from held-out paths.
        training_loader: Shuffled batches used for gradient updates.
        validation_loader: Ordered batches used for validation.
        heldout_loader: Optional ordered batches used for final testing.
    """

    training_data: pd.DataFrame
    validation_data: pd.DataFrame
    heldout_data: pd.DataFrame | None
    feature_scaler: StandardScaler
    position_scaler: StandardScaler
    training_set: PositionWindows
    validation_set: PositionWindows
    heldout_set: PositionWindows | None
    training_loader: DataLoader
    validation_loader: DataLoader
    heldout_loader: DataLoader | None


def load_data(
    path: Path,
    feature_columns: list[str],
    position_columns: list[str],
) -> pd.DataFrame:
    """Read CSV rows containing usable model features and output positions.

    Timestamps are parsed as UTC. Rows with an invalid timestamp, missing path
    identifier, or missing requested feature/position value are dropped.

    Args:
        path: CSV file containing trajectory observations.
        feature_columns: Ordered model input feature columns to read.
        position_columns: Ordered predicted-position columns to read.

    Returns:
        A dataframe sorted by flight path and timestamp, with a reset index.
    """
    # Read the union so columns used as inputs and prediction targets are present.
    model_columns = list(dict.fromkeys([*feature_columns, *position_columns]))
    data = pd.read_csv(
        path,
        usecols=["time", "flight_path_id", *model_columns],
    )
    # Normalize timestamps, remove incomplete observations, and order each path.
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    data = data.dropna(subset=["time", "flight_path_id", *model_columns])
    return data.sort_values(["flight_path_id", "time"], kind="stable").reset_index(
        drop=True
    )


def split_paths(
    data: pd.DataFrame,
    validation_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations by flight path into training and validation frames.

    Args:
        data: Observations containing a ``flight_path_id`` column.
        validation_fraction: Fraction of flight paths assigned to validation.
        random_seed: Seed used to make the group split repeatable.

    Returns:
        Copies of the training and validation rows, in that order.

    Raises:
        ValueError: If fewer than two distinct flight paths are available.
    """
    if data["flight_path_id"].nunique() < 2:
        raise ValueError("At least two flight paths are required.")

    # Split by path ID so observations from one flight cannot leak between sets.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_seed,
    )
    training_indices, validation_indices = next(
        splitter.split(data, groups=data["flight_path_id"])
    )
    return (
        data.iloc[training_indices].copy(),
        data.iloc[validation_indices].copy(),
    )


def create_data_loaders(
    path: Path,
    heldout_path: Path | None,
    feature_columns: list[str],
    position_columns: list[str],
    history_seconds: int,
    forecast_seconds: int,
    batch_size: int,
    validation_fraction: float,
    random_seed: int,
) -> TrainingData:
    """Load a CSV and prepare path-disjoint datasets and DataLoaders.

    Separate feature and position scalers are fitted only on training paths
    and then used for validation and held-out data. Training batches are
    shuffled; validation and held-out batches retain their dataset order.

    Args:
        path: CSV file containing trajectory observations.
        heldout_path: Optional CSV reserved for final testing after training.
        feature_columns: Ordered columns supplied to the model as input.
        position_columns: Ordered columns predicted by the model.
        history_seconds: Number of observed seconds in each input window.
        forecast_seconds: Number of future seconds in each target window.
        batch_size: Number of windows per DataLoader batch.
        validation_fraction: Fraction of distinct flight paths held out.
        random_seed: Seed used for the path-level train/validation split.

    Returns:
        A ``TrainingData`` object containing the split frames, fitted scalers,
        datasets, and DataLoaders. Held-out fields are ``None`` if no held-out
        path is supplied.

    Raises:
        ValueError: If the input has too few flight paths or either split has
            no continuous run long enough to make a complete window.
    """
    data = load_data(path, feature_columns, position_columns)
    training_data, validation_data = split_paths(data, validation_fraction, random_seed)
    # Fit only on training paths so validation and test data cannot leak into
    # the normalization statistics.
    feature_scaler = StandardScaler().fit(
        training_data[feature_columns].to_numpy(dtype=np.float32)
    )
    position_scaler = StandardScaler().fit(
        training_data[position_columns].to_numpy(dtype=np.float32)
    )
    training_set = PositionWindows(
        training_data,
        feature_scaler,
        position_scaler,
        feature_columns,
        position_columns,
        history_seconds,
        forecast_seconds,
    )
    validation_set = PositionWindows(
        validation_data,
        feature_scaler,
        position_scaler,
        feature_columns,
        position_columns,
        history_seconds,
        forecast_seconds,
    )
    heldout_data = None
    heldout_set = None
    heldout_loader = None
    if heldout_path is not None:
        heldout_data = load_data(heldout_path, feature_columns, position_columns)
        # Reuse training scalers; never fit them on held-out test values.
        heldout_set = PositionWindows(
            heldout_data,
            feature_scaler,
            position_scaler,
            feature_columns,
            position_columns,
            history_seconds,
            forecast_seconds,
        )
        heldout_loader = DataLoader(heldout_set, batch_size=batch_size)

    # Shuffle training windows for updates; keep validation deterministic.
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)

    return TrainingData(
        training_data=training_data,
        validation_data=validation_data,
        heldout_data=heldout_data,
        feature_scaler=feature_scaler,
        position_scaler=position_scaler,
        training_set=training_set,
        validation_set=validation_set,
        heldout_set=heldout_set,
        training_loader=training_loader,
        validation_loader=validation_loader,
        heldout_loader=heldout_loader,
    )
