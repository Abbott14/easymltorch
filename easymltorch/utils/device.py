"""Device utility functions for easymltorch."""

import torch


def get_device(device: str | None = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device: Optional device string. If None, automatically selects
                'cuda' if available, otherwise 'cpu'.

    Returns:
        A torch.device object.

    Raises:
        ValueError: If the specified device is not valid.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # Validate the device string
    valid_devices = {"cpu", "cuda", "mps"}
    base_device = device.split(":")[0]

    if base_device not in valid_devices:
        raise ValueError(
            f"Invalid device '{device}'. Must be one of: {', '.join(valid_devices)} "
            f"(optionally with index, e.g., 'cuda:0')"
        )

    # Check if CUDA is requested but not available
    if base_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available")

    # Check if MPS is requested but not available
    if base_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested but MPS is not available")

    return torch.device(device)
