from __future__ import annotations

from typing import Dict, Optional, Tuple

from stable_baselines3.common.evaluation import evaluate_policy


def evaluate(
    model,
    env,
    n_eval_episodes: int = 20,
    deterministic: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    mean_r, std_r = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
        warn=False,
    )
    metrics = {
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
    }
    return float(mean_r), float(std_r), metrics
