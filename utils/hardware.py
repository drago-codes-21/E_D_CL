from __future__ import annotations

import logging

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def detect_torch_device(preferred: str = "auto") -> str:
    """
    Resolve the device for Torch/SentenceTransformers.
    """
    if preferred and preferred.lower() not in {"auto", "auto()"}:
        return preferred

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """
    Provide a human-readable description for logging.
    """
    if torch is None:
        return "cpu (torch not installed)"

    if device == "cuda":
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else "Unknown GPU"
        return f"cuda ({name}, {count} device{'s' if count != 1 else ''})"
    if device == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
