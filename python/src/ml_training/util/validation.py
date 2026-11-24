"""Evaluate trajectory forecasts using scaled loss and displacement metrics.

Validation never updates model parameters. Forecast displacement metrics are
computed in meters after predictions are converted back to the original
latitude, longitude, and altitude units.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data_loader import PositionWindows

EARTH_RADIUS_METERS = 6_371_000


@dataclass(frozen=True)
class ForecastMetrics:
    """Mean-horizon and final-step displacement RMSE values in meters."""

    mean_horizontal_rmse_m: float
    mean_vertical_rmse_m: float
    mean_east_rmse_m: float
    mean_north_rmse_m: float
    final_horizontal_rmse_m: float
    final_vertical_rmse_m: float
    final_east_rmse_m: float
    final_north_rmse_m: float
    selection_score_m: float


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
    validation_set: PositionWindows,
    position_scaler: StandardScaler,
    device: torch.device,
    position_columns: list[str],
    batch_size: int,
) -> tuple[int, ForecastMetrics]:
    """Measure average and final forecast displacement errors in meters.

    Only complete, continuous runs are evaluated. Forecast starts are spaced
    ``forecast_seconds`` apart, producing non-overlapping target windows. The
    dataset yields non-overlapping forecast windows. Position columns must be
    ordered as latitude, longitude, and altitude because those coordinates are
    converted separately.

    Args:
        model: Trained model that predicts standardized positions.
        validation_set: Window dataset providing standardized histories and
            raw-coordinate targets for validation.
        position_scaler: Scaler used to transform predicted positions.
        device: Device on which to execute model inference.
        position_columns: Output position columns in latitude/longitude/
            altitude order.
        batch_size: Maximum number of forecast windows evaluated together.

    Returns:
        A pair containing the number of forecasts and their typed metrics.
        The metrics include mean and final-step east, north, horizontal, and
        vertical RMSE values, plus ``selection_score_m`` (the mean of mean
        horizontal and vertical RMSE).

    Raises:
        ValueError: If no validation run is long enough for the requested
            history and forecast durations.
    """
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

        for (
            scaled_histories,
            actual_positions,
        ) in validation_set.iter_validation_batches(batch_size):
            batch_count = len(scaled_histories)
            history = torch.tensor(scaled_histories, dtype=torch.float32, device=device)
            # Run the model to get standardized predictions. Moving them to CPU
            # allows conversion to NumPy for inverse scaling.
            # Inverse scaling turns the standardized predictions back to the original scale.
            scaled_predictions_array = model(history).cpu().numpy()
            predicted_positions = position_scaler.inverse_transform(
                scaled_predictions_array.reshape(-1, len(position_columns))
            ).reshape(
                batch_count,
                validation_set.forecast_seconds,
                len(position_columns),
            )

            # Compare predictions and actual positions in meters.
            east_error, north_error, vertical_error = calculate_displacement_errors(
                predicted_positions, actual_positions
            )
            # Combine east and north squared errors for horizontal RMSE.
            horizontal_error_squared = np.square(east_error) + np.square(north_error)
            # Accumulate squared errors to calculate horizon-wide RMSE.
            east_squared_error += np.square(east_error).sum()
            north_squared_error += np.square(north_error).sum()
            vertical_squared_error += np.square(vertical_error).sum()
            horizontal_squared_error += horizontal_error_squared.sum()
            # Track final-step errors separately for the endpoint RMSE.
            final_east_squared_error += np.square(east_error[:, -1]).sum()
            final_north_squared_error += np.square(north_error[:, -1]).sum()
            final_vertical_squared_error += np.square(vertical_error[:, -1]).sum()
            final_horizontal_squared_error += horizontal_error_squared[:, -1].sum()
            # Mean metrics cover every forecast position; final metrics one each.
            forecast_count += batch_count
            forecast_position_count += east_error.size

    if forecast_count == 0:
        raise ValueError(
            "No validation examples are available for displacement metrics."
        )

    mean_horizontal_rmse = float(
        np.sqrt(horizontal_squared_error / forecast_position_count)
    )
    mean_vertical_rmse = float(
        np.sqrt(vertical_squared_error / forecast_position_count)
    )
    return forecast_count, ForecastMetrics(
        mean_horizontal_rmse_m=mean_horizontal_rmse,
        mean_vertical_rmse_m=mean_vertical_rmse,
        mean_east_rmse_m=float(np.sqrt(east_squared_error / forecast_position_count)),
        mean_north_rmse_m=float(np.sqrt(north_squared_error / forecast_position_count)),
        final_horizontal_rmse_m=float(
            np.sqrt(final_horizontal_squared_error / forecast_count)
        ),
        final_vertical_rmse_m=float(
            np.sqrt(final_vertical_squared_error / forecast_count)
        ),
        final_east_rmse_m=float(np.sqrt(final_east_squared_error / forecast_count)),
        final_north_rmse_m=float(np.sqrt(final_north_squared_error / forecast_count)),
        selection_score_m=(mean_horizontal_rmse + mean_vertical_rmse) / 2,
    )
