"""Train a small LSTM to predict five minutes of future aircraft positions.

The model uses 120 seconds of history to predict 300 one-second positions directly.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
from time import perf_counter

import torch
from torch import nn

# Support both `python -m ml_training.train_lstm` and direct script execution.
if __package__:
    from .util.data_loader import create_data_loaders
    from .util.device import get_device
    from .util.save_training_results import (
        DisplacementMetrics,
        EpochTrainingResult,
        TrainingResults,
        save_training_results,
    )
    from .util.validation import evaluate_validation_loss, validate_displacement
else:
    from util.data_loader import create_data_loaders
    from util.device import get_device
    from util.save_training_results import (
        DisplacementMetrics,
        EpochTrainingResult,
        TrainingResults,
        save_training_results,
    )
    from util.validation import evaluate_validation_loss, validate_displacement

# Resolve defaults from the repository location, not the current working directory.
ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT_DIR / "out" / "ml" / "data" / "processed" / "adsb_trajectory_data_cleaned.csv"
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
FORECAST_SECONDS = 300  # Number of future one-second positions predicted at once.
HIDDEN_SIZE = 32  # Number of features in the LSTM's hidden state.
BATCH_SIZE = 32  # Number of examples per optimizer update.
EPOCHS = 5  # Full passes over the training examples.
LEARNING_RATE = 0.001  # Adam optimizer step size.
VALIDATION_FRACTION = 0.2  # Fraction of whole flight paths held out for validation.
RANDOM_SEED = 42  # Seed used for repeatable splitting and PyTorch randomness.
FORECAST_MODE = "direct_300_seconds"  # Checkpoint marker for this output format.


class TrajectoryLSTM(nn.Module):
    """Map a history sequence to all positions in a future trajectory.

    The LSTM summarizes the input sequence using its final output. A linear
    layer maps that summary directly to the configured forecast horizon.

    Attributes:
        feature_columns: Ordered input feature names.
        position_columns: Ordered output position names.
        forecast_seconds: Number of future positions predicted by the model.
        lstm: Recurrent layer that encodes the input history.
        output: Linear layer that produces the flattened future trajectory.
    """

    def __init__(
        self,
        feature_columns: list[str],
        position_columns: list[str],
        forecast_seconds: int,
        hidden_size: int,
    ) -> None:
        """Initialize the model architecture.

        Args:
            feature_columns: Ordered input features represented in each row.
            position_columns: Ordered output position columns to predict.
            forecast_seconds: Number of positions to predict directly.
            hidden_size: Number of features in the LSTM's hidden state.
        """
        super().__init__()
        self.feature_columns = feature_columns
        self.position_columns = position_columns
        self.forecast_seconds = forecast_seconds
        # batch_first keeps LSTM inputs and outputs shaped [batch, time, features].
        self.lstm = nn.LSTM(
            input_size=len(feature_columns),
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, forecast_seconds * len(position_columns))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict a batch of future sequences from a batch of histories.

        Args:
            history: Standardized tensor shaped
                ``[batch, history_steps, len(feature_columns)]``.

        Returns:
            Standardized predictions shaped
            ``[batch, forecast_seconds, len(position_columns)]``.
        """
        sequence, _ = self.lstm(history)
        # The final history output summarizes the sequence; the linear layer
        # predicts the entire forecast horizon directly rather than stepwise.
        forecast = self.output(sequence[:, -1])
        # Restore the flattened linear output to [batch, forecast, features].
        return forecast.reshape(
            history.shape[0],
            self.forecast_seconds,
            len(self.position_columns),
        )


def train(
    input_path: Path,
    output_path: Path,
    heldout_path: Path | None = DEFAULT_HELDOUT_INPUT,
) -> None:
    """Train a direct 300-second model and save its best checkpoint."""
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
        FORECAST_SECONDS,
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
        FORECAST_SECONDS,
        HIDDEN_SIZE,
    ).to(device)
    # Use mean squared error for both training and validation.
    loss_function = nn.MSELoss()
    # Use the Adam optimizer to update all model parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lower validation scores indicate better performance, so track the best epoch.
    best_validation_score = float("inf")
    best_epoch = 0
    best_training_mse: float | None = None
    best_validation_loss: float | None = None
    best_forecast_count: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, float] | None = None
    best_displacement_metrics: DisplacementMetrics | None = None
    epoch_history: list[EpochTrainingResult] = []

    for epoch in range(EPOCHS):
        # Set training-specific layer behavior, such as dropout, for this epoch.
        model.train()
        training_loss = 0.0
        for histories, targets in training_loader:
            histories, targets = histories.to(device), targets.to(device)
            # Clear prior gradients, compute the batch loss, then update weights.
            optimizer.zero_grad()
            loss = loss_function(model(histories), targets)
            loss.backward()
            optimizer.step()
            # Weight each batch mean by its size for a true epoch-wide mean.
            training_loss += loss.item() * len(histories)

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
            validation_data,
            feature_scaler,
            position_scaler,
            device,
            HISTORY_SECONDS,
            FORECAST_SECONDS,
            FEATURE_COLUMNS,
            POSITION_COLUMNS,
            BATCH_SIZE,
        )
        training_mse = training_loss / len(training_set)
        displacement_metrics = DisplacementMetrics.from_mapping(forecast_metrics)
        epoch_history.append(
            EpochTrainingResult(
                epoch=epoch + 1,
                training_mse=training_mse,
                validation_mse_standardized=validation_loss,
                validation_forecast_count=forecast_count,
                displacement_metrics=displacement_metrics,
            )
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"train MSE={training_mse:.4f}, "
            f"direct {FORECAST_SECONDS}-second validation MSE={validation_loss:.4f}"
        )
        print(
            f"  Mean over {FORECAST_SECONDS} seconds ({forecast_count} forecasts): "
            f"horizontal RMSE={forecast_metrics['mean_horizontal_rmse_m']:.1f} m, "
            f"vertical RMSE={forecast_metrics['mean_vertical_rmse_m']:.1f} m "
            f"(east={forecast_metrics['mean_east_rmse_m']:.1f} m, "
            f"north={forecast_metrics['mean_north_rmse_m']:.1f} m)"
        )
        print(
            f"  At {FORECAST_SECONDS} seconds: "
            f"horizontal RMSE={forecast_metrics['final_horizontal_rmse_m']:.1f} m, "
            f"vertical RMSE={forecast_metrics['final_vertical_rmse_m']:.1f} m "
            f"(east={forecast_metrics['final_east_rmse_m']:.1f} m, "
            f"north={forecast_metrics['final_north_rmse_m']:.1f} m)"
        )
        print(
            "  Selection score (mean of average horizontal and vertical "
            f"RMSE): {forecast_metrics['selection_score_m']:.1f} m"
        )

        # Keep the epoch with the lowest average horizontal/vertical RMSE score.
        if forecast_metrics["selection_score_m"] < best_validation_score:
            best_validation_score = forecast_metrics["selection_score_m"]
            best_epoch = epoch + 1
            best_training_mse = training_mse
            best_validation_loss = validation_loss
            best_forecast_count = forecast_count
            best_displacement_metrics = displacement_metrics
            # Clone to CPU so later optimizer updates cannot alter this snapshot.
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_validation_metrics = forecast_metrics.copy()

    if (
        best_model_state is None
        or best_validation_metrics is None
        or best_training_mse is None
        or best_validation_loss is None
        or best_forecast_count is None
        or best_displacement_metrics is None
    ):
        raise RuntimeError("Training did not produce a best model checkpoint.")

    # Restore the selected epoch before writing its weights.
    model.load_state_dict(best_model_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            # Store both scalers so inference normalizes inputs and converts
            # predicted positions back to their original units.
            "feature_mean": feature_scaler.mean_.tolist(),
            "feature_scale": feature_scaler.scale_.tolist(),
            "position_mean": position_scaler.mean_.tolist(),
            "position_scale": position_scaler.scale_.tolist(),
            "feature_columns": FEATURE_COLUMNS,
            "position_columns": POSITION_COLUMNS,
            "best_epoch": best_epoch,
            "best_validation_score_m": best_validation_score,
            "best_validation_metrics": best_validation_metrics,
            "forecast_mode": FORECAST_MODE,
        },
        output_path,
    )
    print(
        f"Saved best model to {output_path} "
        f"(epoch {best_epoch}, score={best_validation_score:.1f} m)"
    )

    heldout_examples: int | None = None
    heldout_forecast_count: int | None = None
    heldout_validation_loss: float | None = None
    heldout_displacement_metrics: DisplacementMetrics | None = None
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
        heldout_forecast_count, heldout_metrics = validate_displacement(
            model,
            heldout_data,
            feature_scaler,
            position_scaler,
            device,
            HISTORY_SECONDS,
            FORECAST_SECONDS,
            FEATURE_COLUMNS,
            POSITION_COLUMNS,
            BATCH_SIZE,
        )
        heldout_displacement_metrics = DisplacementMetrics.from_mapping(heldout_metrics)
        print(
            f"Held-out test: standardized MSE={heldout_validation_loss:.4f}, "
            f"selection score={heldout_metrics['selection_score_m']:.1f} m "
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
        forecast_seconds=FORECAST_SECONDS,
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
        best_training_mse=best_training_mse,
        best_validation_mse_standardized=best_validation_loss,
        best_validation_forecast_count=best_forecast_count,
        best_selection_score_m=best_validation_score,
        best_displacement_metrics=best_displacement_metrics,
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
