"""Represent and save training-run results as timestamped JSON."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[4] / "out" / "ml" / "training_results"
)


def _round_float_values(value: object) -> object:
    """Round report floats to four decimal places, preserving other values."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_float_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_float_values(item) for item in value]
    return value


@dataclass(frozen=True)
class DisplacementMetrics:
    """Horizontal and vertical validation RMSE values in meters."""

    mean_horizontal_rmse_m: float
    mean_vertical_rmse_m: float
    mean_east_rmse_m: float
    mean_north_rmse_m: float
    final_horizontal_rmse_m: float
    final_vertical_rmse_m: float
    final_east_rmse_m: float
    final_north_rmse_m: float
    selection_score_m: float

    @classmethod
    def from_mapping(cls, metrics: Mapping[str, float]) -> "DisplacementMetrics":
        """Build the typed metric record from validation's metric dictionary."""
        return cls(
            mean_horizontal_rmse_m=metrics["mean_horizontal_rmse_m"],
            mean_vertical_rmse_m=metrics["mean_vertical_rmse_m"],
            mean_east_rmse_m=metrics["mean_east_rmse_m"],
            mean_north_rmse_m=metrics["mean_north_rmse_m"],
            final_horizontal_rmse_m=metrics["final_horizontal_rmse_m"],
            final_vertical_rmse_m=metrics["final_vertical_rmse_m"],
            final_east_rmse_m=metrics["final_east_rmse_m"],
            final_north_rmse_m=metrics["final_north_rmse_m"],
            selection_score_m=metrics["selection_score_m"],
        )


@dataclass(frozen=True)
class EpochTrainingResult:
    """Losses and actual-distance metrics collected for one epoch."""

    epoch: int
    training_mse: float
    validation_mse_standardized: float
    validation_forecast_count: int
    displacement_metrics: DisplacementMetrics


@dataclass(frozen=True)
class TrainingResults:
    """Configuration, environment, and measured outcomes for a training run.

    ``validation_mse_standardized`` records loss across the full forecast
    horizon. ``best_displacement_metrics`` stores RMSE over that horizon and
    at its final (300-second, with the current configuration) position.
    """

    input_file: Path
    model_checkpoint: Path
    heldout_input_file: Path | None
    heldout_examples: int | None
    heldout_forecast_count: int | None
    heldout_mse_standardized: float | None
    heldout_displacement_metrics: DisplacementMetrics | None
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    python_version: str
    pytorch_version: str
    pytorch_cuda_version: str | None
    device: str
    device_name: str | None
    model_class: str
    epochs_requested: int
    epochs_completed: int
    history_seconds: int
    forecast_seconds: int
    feature_columns: tuple[str, ...]
    position_columns: tuple[str, ...]
    hidden_size: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    random_seed: int
    forecast_mode: str
    optimizer: str
    loss_function: str
    training_flight_paths: int
    validation_flight_paths: int
    training_examples: int
    validation_examples: int
    best_epoch: int
    best_training_mse: float
    best_validation_mse_standardized: float
    best_validation_forecast_count: int
    best_selection_score_m: float
    best_displacement_metrics: DisplacementMetrics
    epoch_history: tuple[EpochTrainingResult, ...]


def save_training_results(results: TrainingResults) -> Path:
    """Write all run results to ``training_results_<UTC timestamp>.json``.

    The timestamped filename keeps reports from multiple runs instead of
    overwriting earlier results.

    Args:
        results: Typed configuration, environment, best-epoch results, and
            per-epoch history to persist.

    Returns:
        The path to the written JSON report.

    Raises:
        OSError: If the report cannot be created or written.
    """
    saved_at = datetime.now(timezone.utc)
    timestamp = saved_at.strftime("%Y%m%d_%H%M%S_%fZ")
    results_path = OUTPUT_DIRECTORY / f"training_results_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    report_data: dict[str, object] = {
        key: _round_float_values(value) for key, value in asdict(results).items()
    }
    report_data["input_file"] = str(results.input_file)
    report_data["model_checkpoint"] = str(results.model_checkpoint)
    if results.heldout_input_file is not None:
        report_data["heldout_input_file"] = str(results.heldout_input_file)
    report_data["report_created_at_utc"] = saved_at.isoformat()
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(report_data, results_file, indent=2)
        results_file.write("\n")
    return results_path
