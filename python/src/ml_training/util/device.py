"""Select the best available PyTorch compute device."""

import torch


def get_device() -> torch.device:
    """Return CUDA, MPS, or CPU, preferring accelerators in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
