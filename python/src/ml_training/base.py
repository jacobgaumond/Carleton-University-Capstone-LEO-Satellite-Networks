import torch

# Setup hardware device

if torch.cuda.is_available():
    device = torch.device("cuda")   # For NVIDIA GPU's
elif torch.backends.mps.is_available():
    device = torch.device("mps")    # For Apple Silicon GPU's
else:
    device = torch.device("cpu")    # Fallback to CPU (not ideal)

torch.set_default_device(device)
