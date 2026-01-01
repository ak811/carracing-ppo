from __future__ import annotations

import platform
from importlib.metadata import version

import torch


def get_system_info() -> str:
    """
    Returns a human-readable block of versions useful for reproducibility.
    """
    lines = [
        f"Python Version: {platform.python_version()}",
        f"Torch Version: {version('torch')}",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Version: {torch.version.cuda}",
    ]
    for pkg in ["gymnasium", "numpy", "stable_baselines3", "opencv-python"]:
        try:
            lines.append(f"{pkg} Version: {version(pkg)}")
        except Exception:
            pass
    return "\n".join(lines)
