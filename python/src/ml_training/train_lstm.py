"""Train a small LSTM to predict five minutes of future aircraft positions.

The model uses 120 seconds of history to predict 300 one-second positions directly.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
from time import perf_counter, time

from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

# Support both `python -m ml_training.train_lstm` and direct script execution.
if __package__:
    from .util.data_loader import PositionWindows, create_data_loaders
    from .util.device import get_device
    from .util.save_training_results import (
        EpochTrainingResult,
        TrainingResults,
        save_training_results,
    )
    from .util.validation import (
        ForecastMetrics,
        evaluate_validation_loss,
        validate_displacement,
    )
else:
    from util.data_loader import PositionWindows, create_data_loaders
    from util.device import get_device
    from util.save_training_results import (
        EpochTrainingResult,
        TrainingResults,
        save_training_results,
    )
    from util.validation import (
        ForecastMetrics,
        evaluate_validation_loss,
        validate_displacement,
    )

# Resolve defaults from the repository location, not the current working directory.
ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT_DIR
    / "out"
    / "ml"
    / "data"
    / "processed"
    / "adsb_trajectory_data_cleaned_strict_interpolated.csv"
)
DEFAULT_OUTPUT = ROOT_DIR / "out" / "ml" / "models" / "aircraft_lstm.pt"
DEFAULT_HELDOUT_INPUT = (
    ROOT_DIR
    / "out"
    / "ml"
    / "data"
    / "processed"
    / "held_out_adsb_trajectory_data_cleaned.csv"
)
# Inputs can include non-position features; output positions remain separate.
FEATURE_COLUMNS = ["lat", "lon", "geoaltitude", "velocity", "heading"]
POSITION_COLUMNS = ["lat", "lon", "geoaltitude"]
HISTORY_SECONDS = 120  # Number of observed one-second positions per input.
FORECAST_HORIZON_SECONDS = 300  # How far into the future the model predicts.
FORECAST_INTERVAL_SECONDS = 10  # Time between predicted positions.
HIDDEN_SIZE = 32  # Number of features in the LSTM's hidden state.
BATCH_SIZE = 32  # Number of examples per optimizer update.
EPOCHS = 5  # Full passes over the training examples.
LEARNING_RATE = 0.001  # Adam optimizer step size.
VALIDATION_FRACTION = 0.2  # Fraction of whole flight paths held out for validation.
RANDOM_SEED = 42  # Seed used for repeatable splitting and PyTorch randomness.
FORECAST_MODE = "direct_300_seconds_every_10_seconds"


class TrajectoryLSTM(nn.Module):
    """Map a history sequence to all positions in a future trajectory.

    The LSTM summarizes the input sequence using its final output. A linear
    layer maps that summary directly to the configured forecast horizon.

    Attributes:
        feature_columns: Ordered input feature names.
        position_columns: Ordered output position names.
        forecast_horizon_seconds: Total time covered by the forecast.
        forecast_interval_seconds: Seconds between predicted positions.
        forecast_steps: Number of positions predicted in the horizon.
        lstm: Recurrent layer that encodes the input history.
        output: Linear layer that produces the flattened future trajectory.
    """

    def __init__(
        self,
        feature_columns: list[str],
        position_columns: list[str],
        forecast_horizon_seconds: int,
        forecast_interval_seconds: int,
        hidden_size: int,
    ) -> None:
        """Initialize the model architecture.

        Args:
            feature_columns: Ordered input features represented in each row.
            position_columns: Ordered output position columns to predict.
            forecast_horizon_seconds: Total future duration represented.
            forecast_interval_seconds: Seconds between predicted positions.
            hidden_size: Number of features in the LSTM's hidden state.
        """
        super().__init__()
        if forecast_horizon_seconds <= 0 or forecast_interval_seconds <= 0:
            raise ValueError("Forecast horizon and interval must be positive.")
        if forecast_horizon_seconds % forecast_interval_seconds != 0:
            raise ValueError(
                "Forecast horizon must be divisible by the forecast interval."
            )

        self.feature_columns = feature_columns
        self.position_columns = position_columns
        self.forecast_horizon_seconds = forecast_horizon_seconds
        self.forecast_interval_seconds = forecast_interval_seconds
        self.forecast_steps = forecast_horizon_seconds // forecast_interval_seconds
        # batch_first keeps LSTM inputs and outputs shaped [batch, time, features].
        self.lstm = nn.LSTM(
            input_size=len(feature_columns),
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(
            hidden_size,
            self.forecast_steps * len(position_columns),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict a batch of future sequences from a batch of histories.

        Args:
            history: Standardized tensor shaped
                ``[batch, history_steps, len(feature_columns)]``.

        Returns:
            Standardized predictions shaped
            ``[batch, forecast_steps, len(position_columns)]``, with output
            steps spaced ``forecast_interval_seconds`` apart through the
            configured forecast horizon.

        Notes:
            Each batch item is processed as an independent history window.
            Since no initial hidden/cell state is provided, the LSTM starts
            from zero state; its final state is not carried to later calls.
        """
        sequence, _ = self.lstm(history)
        # The final history output summarizes the sequence; the linear layer
        # predicts the entire forecast horizon directly rather than stepwise.
        forecast = self.output(sequence[:, -1])
        # Restore the flattened linear output to [batch, forecast, features].
        return forecast.reshape(
            history.shape[0],
            self.forecast_steps,
            len(self.position_columns),
        )


def train_one_epoch(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    validation_set: PositionWindows,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    position_scaler: StandardScaler,
    position_columns: list[str],
    validation_batch_size: int,
    epoch: int,
) -> EpochTrainingResult:
    """Train once and evaluate validation loss and forecast displacement."""
    model.train()
    total_loss = 0.0
    example_count = 0
    total_batches = len(training_loader)
    progress_width = 30
    start = time()

    for batch_index, (histories, targets) in enumerate(training_loader, start=1):
        histories, targets = histories.to(device), targets.to(device)
        # Clear prior gradients before calculating this batch's update.
        optimizer.zero_grad()
        loss = loss_function(model(histories), targets)
        loss.backward()
        optimizer.step()

        # Weight batch means by their example counts for an epoch-wide mean.
        batch_size = len(histories)
        total_loss += loss.item() * batch_size
        example_count += batch_size

        if batch_index % 1000 == 0:  # Print progress every 1000 batches
            completed_width = int(progress_width * batch_index / total_batches)
            progress_bar = "=" * completed_width + " " * (
                progress_width - completed_width
            )
            current_mse = total_loss / example_count
            progress_per_s = batch_index / (time() - start)
            eta_s = (
                (total_batches - batch_index) / progress_per_s
                if progress_per_s > 0
                else 0
            )
            print(
                f"\rTraining [{progress_bar}] {batch_index}/{total_batches} "
                f"MSE: {current_mse:.4f}, ETA: {eta_s:.0f} s    ",  # whitespace so the printover doesnt leave trailing characters
                end="",
                flush=True,
            )

    if example_count == 0:
        raise ValueError("Training loader produced no examples for this epoch.")

    print()
    training_mse = total_loss / example_count

    # Evaluate both scaled prediction loss and displacement in meters.
    validation_loss = evaluate_validation_loss(
        model,
        validation_loader,
        loss_function,
        device,
        len(validation_set),
    )
    forecast_count, forecast_metrics = validate_displacement(
        model,
        validation_set,
        position_scaler,
        device,
        position_columns,
        validation_batch_size,
    )
    return EpochTrainingResult(
        epoch=epoch,
        training_mse=training_mse,
        validation_mse_standardized=validation_loss,
        validation_forecast_count=forecast_count,
        displacement_metrics=forecast_metrics,
    )


def print_epoch_result(
    result: EpochTrainingResult,
    total_epochs: int,
    forecast_horizon_seconds: int,
    forecast_interval_seconds: int,
) -> None:
    """Print training loss and validation displacement metrics for one epoch."""
    metrics = result.displacement_metrics
    print(
        f"Epoch {result.epoch}/{total_epochs}: "
        f"train MSE={result.training_mse:.4f}, "
        f"direct {forecast_horizon_seconds}-second validation "
        f"MSE={result.validation_mse_standardized:.4f}"
    )
    print(
        f"  Mean over {forecast_horizon_seconds // forecast_interval_seconds} "
        f"forecast points sampled every {forecast_interval_seconds} seconds "
        f"({result.validation_forecast_count} forecasts): "
        f"horizontal RMSE={metrics.mean_horizontal_rmse_m:.1f} m, "
        f"vertical RMSE={metrics.mean_vertical_rmse_m:.1f} m "
        f"(east={metrics.mean_east_rmse_m:.1f} m, "
        f"north={metrics.mean_north_rmse_m:.1f} m)"
    )
    print(
        f"  At {forecast_horizon_seconds} seconds: "
        f"horizontal RMSE={metrics.final_horizontal_rmse_m:.1f} m, "
        f"vertical RMSE={metrics.final_vertical_rmse_m:.1f} m "
        f"(east={metrics.final_east_rmse_m:.1f} m, "
        f"north={metrics.final_north_rmse_m:.1f} m)"
    )
    print(
        "  Selection score (mean of average horizontal and vertical "
        f"RMSE): {metrics.selection_score_m:.1f} m"
    )


def save_best_model(
    model: nn.Module,
    output_path: Path,
    best_model_state: dict[str, torch.Tensor],
    feature_scaler: StandardScaler,
    position_scaler: StandardScaler,
    feature_columns: list[str],
    position_columns: list[str],
    best_epoch: EpochTrainingResult,
    forecast_horizon_seconds: int,
    forecast_interval_seconds: int,
    forecast_mode: str,
) -> None:
    """Restore and save the best checkpoint with its scaling metadata."""
    if (
        feature_scaler.mean_ is None
        or feature_scaler.scale_ is None
        or position_scaler.mean_ is None
        or position_scaler.scale_ is None
    ):
        raise RuntimeError("Training scalers are not fitted.")

    model.load_state_dict(best_model_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "feature_mean": feature_scaler.mean_.tolist(),
            "feature_scale": feature_scaler.scale_.tolist(),
            "position_mean": position_scaler.mean_.tolist(),
            "position_scale": position_scaler.scale_.tolist(),
            "feature_columns": feature_columns,
            "position_columns": position_columns,
            "best_epoch": best_epoch.epoch,
            "best_validation_score_m": (
                best_epoch.displacement_metrics.selection_score_m
            ),
            "best_validation_metrics": asdict(best_epoch.displacement_metrics),
            "forecast_horizon_seconds": forecast_horizon_seconds,
            "forecast_interval_seconds": forecast_interval_seconds,
            "forecast_steps": (forecast_horizon_seconds // forecast_interval_seconds),
            "forecast_mode": forecast_mode,
        },
        output_path,
    )
    print(
        f"Saved best model to {output_path} "
        f"(epoch {best_epoch.epoch}, "
        f"score={best_epoch.displacement_metrics.selection_score_m:.1f} m)"
    )


def train(
    input_path: Path,
    output_path: Path,
    heldout_path: Path | None = DEFAULT_HELDOUT_INPUT,
) -> None:
    """Train a 300-second trajectory model with 10-second output intervals."""
    run_started_at = datetime.now(timezone.utc)
    run_started = perf_counter()
    # Seed weight initialization and training-window shuffling.
    torch.manual_seed(RANDOM_SEED)
    prepared_data = create_data_loaders(
        input_path,
        heldout_path,
        FEATURE_COLUMNS,
        POSITION_COLUMNS,
        HISTORY_SECONDS,
        FORECAST_HORIZON_SECONDS,
        FORECAST_INTERVAL_SECONDS,
        BATCH_SIZE,
        VALIDATION_FRACTION,
        RANDOM_SEED,
    )
    training_data = prepared_data.training_data
    validation_data = prepared_data.validation_data
    heldout_data = prepared_data.heldout_data
    feature_scaler = prepared_data.feature_scaler
    position_scaler = prepared_data.position_scaler
    training_set = prepared_data.training_set
    validation_set = prepared_data.validation_set
    heldout_set = prepared_data.heldout_set
    training_loader = prepared_data.training_loader
    validation_loader = prepared_data.validation_loader
    heldout_loader = prepared_data.heldout_loader

    # Checkpoint scaling values are needed later to convert forecasts to units.
    if (
        feature_scaler.mean_ is None
        or feature_scaler.scale_ is None
        or position_scaler.mean_ is None
        or position_scaler.scale_ is None
    ):
        raise RuntimeError("Training scalers are not fitted.")

    device = get_device()

    # Create a new model and optimizer for this training run.
    model = TrajectoryLSTM(
        FEATURE_COLUMNS,
        POSITION_COLUMNS,
        FORECAST_HORIZON_SECONDS,
        FORECAST_INTERVAL_SECONDS,
        HIDDEN_SIZE,
    ).to(device)

    # Use mean squared error for both training and validation.
    loss_function = nn.L1Loss()

    # Use the Adam optimizer to update all model parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lower validation scores indicate better performance.
    best_epoch: EpochTrainingResult | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    epoch_history: list[EpochTrainingResult] = []

    for epoch in range(EPOCHS):
        epoch_result = train_one_epoch(
            model,
            training_loader,
            validation_loader,
            validation_set,
            loss_function,
            optimizer,
            device,
            position_scaler,
            POSITION_COLUMNS,
            BATCH_SIZE,
            epoch + 1,
        )
        epoch_history.append(epoch_result)
        print_epoch_result(
            epoch_result,
            EPOCHS,
            FORECAST_HORIZON_SECONDS,
            FORECAST_INTERVAL_SECONDS,
        )

        # Keep the epoch with the lowest average horizontal/vertical RMSE score.
        if (
            best_epoch is None
            or epoch_result.displacement_metrics.selection_score_m
            < best_epoch.displacement_metrics.selection_score_m
        ):
            best_epoch = epoch_result
            # Clone to CPU so later optimizer updates cannot alter this snapshot.
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_model_state is None or best_epoch is None:
        raise RuntimeError("Training did not produce a best model checkpoint.")

    save_best_model(
        model,
        output_path,
        best_model_state,
        feature_scaler,
        position_scaler,
        FEATURE_COLUMNS,
        POSITION_COLUMNS,
        best_epoch,
        FORECAST_HORIZON_SECONDS,
        FORECAST_INTERVAL_SECONDS,
        FORECAST_MODE,
    )

    heldout_examples: int | None = None
    heldout_forecast_count: int | None = None
    heldout_validation_loss: float | None = None
    heldout_displacement_metrics: ForecastMetrics | None = None
    if heldout_loader is not None:
        if heldout_data is None or heldout_set is None:
            raise RuntimeError("Held-out loader is missing its source data or dataset.")
        # Evaluate only after restoring the best validation-selected weights.
        heldout_examples = len(heldout_set)
        heldout_validation_loss = evaluate_validation_loss(
            model,
            heldout_loader,
            loss_function,
            device,
            heldout_examples,
        )
        heldout_forecast_count, heldout_displacement_metrics = validate_displacement(
            model,
            heldout_set,
            position_scaler,
            device,
            POSITION_COLUMNS,
            BATCH_SIZE,
        )
        print(
            f"Held-out test: standardized MSE={heldout_validation_loss:.4f}, "
            f"selection score={heldout_displacement_metrics.selection_score_m:.1f} m "
            f"({heldout_forecast_count} forecasts)"
        )

    run_completed_at = datetime.now(timezone.utc)
    results = TrainingResults(
        input_file=input_path.resolve(),
        model_checkpoint=output_path.resolve(),
        heldout_input_file=(
            heldout_path.resolve() if heldout_path is not None else None
        ),
        heldout_examples=heldout_examples,
        heldout_forecast_count=heldout_forecast_count,
        heldout_mse_standardized=heldout_validation_loss,
        heldout_displacement_metrics=heldout_displacement_metrics,
        started_at_utc=run_started_at.isoformat(),
        completed_at_utc=run_completed_at.isoformat(),
        duration_seconds=perf_counter() - run_started,
        python_version=platform.python_version(),
        pytorch_version=str(torch.__version__),
        pytorch_cuda_version=torch.version.cuda,
        device=str(device),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        model_class=type(model).__name__,
        epochs_requested=EPOCHS,
        epochs_completed=len(epoch_history),
        history_seconds=HISTORY_SECONDS,
        forecast_seconds=FORECAST_HORIZON_SECONDS,
        forecast_interval_seconds=FORECAST_INTERVAL_SECONDS,
        feature_columns=tuple(FEATURE_COLUMNS),
        position_columns=tuple(POSITION_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        validation_fraction=VALIDATION_FRACTION,
        random_seed=RANDOM_SEED,
        forecast_mode=FORECAST_MODE,
        optimizer=type(optimizer).__name__,
        loss_function=type(loss_function).__name__,
        training_flight_paths=int(training_data["flight_path_id"].nunique()),
        validation_flight_paths=int(validation_data["flight_path_id"].nunique()),
        training_examples=len(training_set),
        validation_examples=len(validation_set),
        best_epoch=best_epoch,
        epoch_history=tuple(epoch_history),
    )
    results_path = save_training_results(results)
    print(f"Saved training results to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--heldout-input",
        type=Path,
        default=DEFAULT_HELDOUT_INPUT,
        help="Separate CSV used only for testing the selected best epoch.",
    )
    args = parser.parse_args()
    train(args.input, args.output, args.heldout_input)


if __name__ == "__main__":
    main()
