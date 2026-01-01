from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_evaluations_npz(npz_path: str, out_path: str, title: Optional[str] = None) -> bool:
    """
    Plot SB3 EvalCallback output (evaluations.npz).
    Saves a single curve (mean ± std across eval episodes).
    Returns False if file not found/invalid.
    """
    if not os.path.exists(npz_path):
        return False

    data = np.load(npz_path)
    if "timesteps" not in data or "results" not in data:
        return False

    timesteps = data["timesteps"]
    results = data["results"]  # shape: [n_evals, n_eval_episodes]
    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_results)
    plt.fill_between(timesteps, mean_results - std_results, mean_results + std_results, alpha=0.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(title or "Evaluation Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True
