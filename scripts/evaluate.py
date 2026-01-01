from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from stable_baselines3 import PPO

from carracing_ppo.envs.carracing import make_carracing_vec_env
from carracing_ppo.training.eval_sb3 import evaluate
from carracing_ppo.utils.io import load_config


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CarRacing PPO model (SB3).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .zip model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(str(cfg.get("device", "auto")))

    env = make_carracing_vec_env(
        env_id=cfg["env_name"],
        n_envs=1,
        seed=int(cfg.get("seed", 42)) + 123,
        grayscale=bool(cfg.get("grayscale", True)),
        resize=cfg.get("resize", 84),
        frame_stack=int(cfg.get("frame_stack", 4)),
        render_mode=None,
    )

    model = PPO.load(args.checkpoint, env=env, device=device)
    mean_r, std_r, metrics = evaluate(
        model=model,
        env=env,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic or bool(cfg.get("deterministic_eval", True)),
    )

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    env.close()


if __name__ == "__main__":
    main()
