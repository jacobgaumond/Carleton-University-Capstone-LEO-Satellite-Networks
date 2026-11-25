"""Load checkpoints and forecast directly from recent trajectory positions.

The model architecture is defined in ``train_lstm`` and is supplied to the
checkpoint loader as an initialized PyTorch module.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_model(
    path: Path,
    model: nn.Module,
    forecast_mode: str,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a checkpoint and its input-feature and output-position scalers.

    Args:
        path: Path to the serialized training checkpoint.
        model: Initialized architecture matching the checkpoint's state dict.
        forecast_mode: Expected checkpoint mode identifier.

    Returns:
        An evaluation-mode model, feature means/scales, and position
        means/scales, each represented by one-dimensional arrays.

    Raises:
        ValueError: If the checkpoint's forecast mode does not match the
            requested mode or required scaling values are missing.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("forecast_mode") != forecast_mode:
        raise ValueError(
            f"Checkpoint forecast mode must be {forecast_mode!r}. "
            "Train a compatible model to create a matching checkpoint."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Older checkpoints used one scaler for both inputs and predicted positions.
    feature_mean = checkpoint.get("feature_mean", checkpoint.get("mean"))
    feature_scale = checkpoint.get("feature_scale", checkpoint.get("scale"))
    position_mean = checkpoint.get("position_mean", checkpoint.get("mean"))
    position_scale = checkpoint.get("position_scale", checkpoint.get("scale"))
    if any(
        values is None
        for values in (feature_mean, feature_scale, position_mean, position_scale)
    ):
        raise ValueError("Checkpoint is missing input or output scaling values.")

    return (
        model,
        np.asarray(feature_mean, dtype=np.float32),
        np.asarray(feature_scale, dtype=np.float32),
        np.asarray(position_mean, dtype=np.float32),
        np.asarray(position_scale, dtype=np.float32),
    )


def forecast(
    model: nn.Module,
    recent_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    position_mean: np.ndarray,
    position_scale: np.ndarray,
    feature_columns: list[str],
    position_columns: list[str],
    history_seconds: int,
) -> np.ndarray:
    """Forecast positions from recent rows of model input features.

    Input features are standardized with their training statistics before
    inference. Predicted positions are converted back using output statistics.

    Args:
        model: Loaded trajectory model with a ``forward`` method accepting a
            batched history tensor.
        recent_features: Unscaled inputs shaped
            ``[history_seconds, len(feature_columns)]``.
        feature_mean: Training mean for each input feature.
        feature_scale: Training scale for each input feature.
        position_mean: Training mean for each output position.
        position_scale: Training scale for each output position.
        feature_columns: Ordered input feature names.
        position_columns: Ordered output position names.
        history_seconds: Required number of recent observations.

    Returns:
        Unscaled future positions shaped
        ``[forecast_steps, len(position_columns)]``, where ``forecast_steps``
        is the model's configured output count. Rows follow the model's
        configured forecast interval; timestamps are not included.

    Raises:
        ValueError: If ``recent_features`` does not have the required shape.
    """
    if recent_features.shape != (history_seconds, len(feature_columns)):
        raise ValueError(
            f"Expected {history_seconds} recent feature rows with columns "
            f"{feature_columns}."
        )

    device = next(model.parameters()).device
    # Match the feature normalization used during training.
    history = torch.tensor(
        (recent_features - feature_mean) / feature_scale,
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        # Add a batch dimension for inference, then remove it from the result.
        scaled_predictions = model(history.unsqueeze(0))[0].cpu().numpy()
    if scaled_predictions.ndim != 2 or scaled_predictions.shape[1] != len(
        position_columns
    ):
        raise ValueError(
            "Model output does not match the configured position columns "
            f"{position_columns}."
        )
    return scaled_predictions * position_scale + position_mean
