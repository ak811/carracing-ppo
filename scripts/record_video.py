from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder

from carracing_ppo.envs.carracing import make_carracing_vec_env
from carracing_ppo.utils.io import load_config


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main():
    parser = argparse.ArgumentParser(description="Record a rollout video from a CarRacing PPO checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .zip model")
    parser.add_argument("--out_dir", default="assets/videos")
    parser.add_argument("--video_length", type=int, default=20000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(str(cfg.get("device", "auto")))
    os.makedirs(args.out_dir, exist_ok=True)

    # Need render_mode for video frames
    env = make_carracing_vec_env(
        env_id=cfg["env_name"],
        n_envs=1,
        seed=int(cfg.get("seed", 42)),
        grayscale=bool(cfg.get("grayscale", True)),
        resize=cfg.get("resize", 84),
        frame_stack=int(cfg.get("frame_stack", 4)),
        render_mode="rgb_array",
    )

    model = PPO.load(args.checkpoint, env=env, device=device)

    env = VecVideoRecorder(
        env,
        video_folder=args.out_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=args.video_length,
        name_prefix="carracing_ppo",
    )

    obs = env.reset()
    for _ in range(args.video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones.any():
            break

    env.close()
    print(f"Recorded video to: {args.out_dir}")


if __name__ == "__main__":
    main()
