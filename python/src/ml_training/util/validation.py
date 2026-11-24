"""Evaluate trajectory forecasts using scaled loss and displacement metrics.

Validation never updates model parameters. Forecast displacement metrics are
computed in meters after predictions are converted back to the original
latitude, longitude, and altitude units.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

EARTH_RADIUS_METERS = 6_371_000


def evaluate_validation_loss(
    model: nn.Module,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    validation_size: int,
) -> float:
    """Calculate the average loss over validation batches.

    The model is put in evaluation mode and gradients are disabled. Batch
    losses are weighted by batch size before the total is divided by the
    number of validation examples.

    Args:
        model: Trained trajectory model to evaluate.
        validation_loader: Batches of standardized histories and targets.
        loss_function: Loss function used during training.
        device: Device on which to run each batch.
        validation_size: Total number of examples in the validation dataset.

    Returns:
        Mean loss per validation example.
    """
    model.eval()  # Set the model to evaluation mode, which affects layers like dropout and batch normalization.
    validation_loss = 0.0
    with torch.no_grad():  # Disable gradient computation for efficiency since we are only evaluating the model.
        for histories, targets in validation_loader:
            # Move the input data and targets to the specified device (CPU or GPU) for computation.
            histories, targets = histories.to(device), targets.to(device)
            # Compute the loss for the current batch and accumulate it, weighted by the number of examples in the batch.
            validation_loss += loss_function(model(histories), targets).item() * len(
                histories
            )

    # Return the average loss per validation example by dividing the total accumulated loss by the total number of validation examples.
    return validation_loss / validation_size


def calculate_displacement_errors(
    predicted_positions: np.ndarray,
    actual_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate east, north, and vertical errors for position batches.

    Both inputs are shaped ``[forecasts, forecast_steps, 3]`` with coordinates
    ordered as latitude in degrees, longitude in degrees, and altitude in
    meters.

    Data structure:
        predicted_positions = np.array([
            [[lat1, lon1, alt1], [lat2, lon2, alt2], ...],  # Forecast 1
            [[lat1, lon1, alt1], [lat2, lon2, alt2], ...],  # Forecast 2
            ...
        ])
        actual_positions = np.array([
            [[lat1, lon1, alt1], [lat2, lon2, alt2], ...],  # Forecast 1
            [[lat1, lon1, alt1], [lat2, lon2, alt2], ...],  # Forecast 2
            ...
        ])

    Args:
        predicted_positions: Forecast positions in their original units.
        actual_positions: Corresponding observed positions in the same units.

    Returns:
        East, north, and vertical displacement errors in meters, each shaped
        ``[forecasts, forecast_steps]``.
    """
    predicted_latitude = predicted_positions[:, :, 0]
    actual_latitude = actual_positions[:, :, 0]
    predicted_longitude = predicted_positions[:, :, 1]
    actual_longitude = actual_positions[:, :, 1]
    predicted_altitude = predicted_positions[:, :, 2]
    actual_altitude = actual_positions[:, :, 2]

    # Longitude is cyclic: wrapping the difference into [-180, 180) ensures
    # crossing the antimeridian uses the shorter angular distance.
    longitude_difference = np.deg2rad(
        (predicted_longitude - actual_longitude + 180) % 360 - 180
    )
    # Use the midpoint latitude as a symmetric representative for the local
    # east-west scale; this is an equirectangular approximation, not a geodesic.
    mean_latitude = np.deg2rad((predicted_latitude + actual_latitude) / 2)

    # A longitude degree spans less distance near the poles. The parallel at
    # latitude φ has radius R*cos(φ), so scale the longitude angle by it.
    east_error = EARTH_RADIUS_METERS * np.cos(mean_latitude) * longitude_difference

    # Latitude follows a meridian whose radius is R, so its angular difference
    # converts directly to north-south distance without a cosine adjustment.
    north_error = EARTH_RADIUS_METERS * np.deg2rad(predicted_latitude - actual_latitude)

    # Altitude is already measured in meters; no angular conversion is needed.
    vertical_error = predicted_altitude - actual_altitude
    return east_error, north_error, vertical_error


def validate_displacement(
    model: nn.Module,
    data: pd.DataFrame,
    feature_scaler: StandardScaler,
    position_scaler: StandardScaler,
    device: torch.device,
    history_seconds: int,
    forecast_seconds: int,
    feature_columns: list[str],
    position_columns: list[str],
    batch_size: int,
) -> tuple[int, dict[str, float]]:
    """Measure average and final forecast displacement errors in meters.

    Only complete, continuous runs are evaluated. Forecast starts are spaced
    ``forecast_seconds`` apart, producing non-overlapping target windows. The
    supplied position columns must be ordered as latitude, longitude, and
    altitude because those coordinates are converted separately.

    Args:
        model: Trained model that predicts standardized positions.
        data: Validation observations with timestamps and flight path IDs.
        feature_scaler: Scaler used to transform model input features.
        position_scaler: Scaler used to transform predicted positions.
        device: Device on which to execute model inference.
        history_seconds: Number of history positions required per forecast.
        forecast_seconds: Number of future positions predicted at each start.
        feature_columns: Ordered input feature columns.
        position_columns: Output position columns in latitude/longitude/
            altitude order.
        batch_size: Maximum number of forecast windows evaluated together.

    Returns:
        A pair containing the number of forecasts and a metrics dictionary.
        The dictionary includes mean and final-step east, north, horizontal,
        and vertical RMSE values, plus ``selection_score_m`` (the mean of mean
        horizontal and vertical RMSE).

    Raises:
        ValueError: If no validation run is long enough for the requested
            history and forecast durations.
    """
    required_rows = history_seconds + forecast_seconds
    forecast_count = 0
    east_squared_error = 0.0
    north_squared_error = 0.0
    vertical_squared_error = 0.0
    horizontal_squared_error = 0.0
    final_east_squared_error = 0.0
    final_north_squared_error = 0.0
    final_vertical_squared_error = 0.0
    final_horizontal_squared_error = 0.0
    forecast_position_count = 0

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency since we are only evaluating the model.

        # Iterate over each unique flight path in the validation data
        for _, path in data.groupby("flight_path_id", sort=False):
            elapsed_seconds = path["time"].diff().dt.total_seconds().to_numpy()
            features = path[feature_columns].to_numpy(dtype=np.float32)
            positions = path[position_columns].to_numpy(dtype=np.float32)
            # Start a new run at every interval that is not exactly one second.
            breaks = np.flatnonzero(elapsed_seconds[1:] != 1.0) + 1

            for feature_run, position_run in zip(
                np.split(features, breaks),
                np.split(positions, breaks),
            ):
                last_start = len(feature_run) - required_rows
                if last_start < 0:
                    continue

                # Step by the forecast horizon to avoid overlapping targets.
                starts = np.arange(0, last_start + 1, forecast_seconds, dtype=np.int64)

                # Process starts in batches to limit memory use.
                for batch_start in range(0, len(starts), batch_size):
                    batch_starts = starts[batch_start : batch_start + batch_size]
                    # Stack each history and its matching actual forecast.
                    histories = np.stack(
                        [
                            feature_run[start : start + history_seconds]
                            for start in batch_starts
                        ]
                    )
                    actual_positions = np.stack(
                        [
                            position_run[
                                start
                                + history_seconds : start
                                + history_seconds
                                + forecast_seconds
                            ]
                            for start in batch_starts
                        ]
                    )

                    # StandardScaler accepts 2D rows, so flatten and restore
                    # the batch and time dimensions around the feature axis.
                    scaled_histories = feature_scaler.transform(
                        histories.reshape(-1, len(feature_columns))
                    ).reshape(
                        len(batch_starts),
                        history_seconds,
                        len(feature_columns),
                    )
                    # Restore [batch, history, features] for the LSTM.
                    history = torch.tensor(
                        scaled_histories, dtype=torch.float32, device=device
                    )
                    # Run the model to get standardized predictions. Moving
                    # them to CPU allows conversion to NumPy for scaler use.
                    scaled_predictions_array = model(history).cpu().numpy()
                    # Flatten [batch, forecast, features] for StandardScaler,
                    # then restore the forecast batch shape after conversion.
                    predicted_positions = position_scaler.inverse_transform(
                        scaled_predictions_array.reshape(-1, len(position_columns))
                    ).reshape(
                        len(batch_starts),
                        forecast_seconds,
                        len(position_columns),
                    )

                    # Compare predictions and actual positions in meters.
                    east_error, north_error, vertical_error = (
                        calculate_displacement_errors(
                            predicted_positions, actual_positions
                        )
                    )
                    # Combine east and north squared errors for horizontal RMSE.
                    horizontal_error_squared = np.square(east_error) + np.square(
                        north_error
                    )
                    # Accumulate squared errors to calculate horizon-wide RMSE.
                    east_squared_error += np.square(east_error).sum()
                    north_squared_error += np.square(north_error).sum()
                    vertical_squared_error += np.square(vertical_error).sum()
                    horizontal_squared_error += horizontal_error_squared.sum()
                    # Track final-step errors separately for the endpoint RMSE.
                    final_east_squared_error += np.square(east_error[:, -1]).sum()
                    final_north_squared_error += np.square(north_error[:, -1]).sum()
                    final_vertical_squared_error += np.square(
                        vertical_error[:, -1]
                    ).sum()
                    final_horizontal_squared_error += horizontal_error_squared[
                        :, -1
                    ].sum()
                    # Count forecasts separately from forecast positions: mean
                    # metrics cover all horizon steps, final metrics one each.
                    forecast_count += len(batch_starts)
                    forecast_position_count += east_error.size

    if forecast_count == 0:
        raise ValueError(
            f"No validation flight has a continuous {forecast_seconds}-second forecast "
            f"after the {history_seconds}-second history."
        )

    mean_horizontal_rmse = float(
        np.sqrt(horizontal_squared_error / forecast_position_count)
    )
    mean_vertical_rmse = float(
        np.sqrt(vertical_squared_error / forecast_position_count)
    )
    return forecast_count, {
        "mean_horizontal_rmse_m": mean_horizontal_rmse,
        "mean_vertical_rmse_m": mean_vertical_rmse,
        "mean_east_rmse_m": float(
            np.sqrt(east_squared_error / forecast_position_count)
        ),
        "mean_north_rmse_m": float(
            np.sqrt(north_squared_error / forecast_position_count)
        ),
        "final_horizontal_rmse_m": float(
            np.sqrt(final_horizontal_squared_error / forecast_count)
        ),
        "final_vertical_rmse_m": float(
            np.sqrt(final_vertical_squared_error / forecast_count)
        ),
        "final_east_rmse_m": float(np.sqrt(final_east_squared_error / forecast_count)),
        "final_north_rmse_m": float(
            np.sqrt(final_north_squared_error / forecast_count)
        ),
        "selection_score_m": (mean_horizontal_rmse + mean_vertical_rmse) / 2,
    }
