from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from carracing_ppo.envs.carracing import make_carracing_vec_env
from carracing_ppo.training.eval_sb3 import evaluate
from carracing_ppo.utils.io import dump_config, get_run_dir
from carracing_ppo.utils.plotting import plot_evaluations_npz
from carracing_ppo.utils.system_info import get_system_info


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train(cfg: Dict[str, Any]) -> str:
    """
    Train SB3 PPO on CarRacing and write outputs into a run directory.

    Returns: run_dir
    """
    device = _resolve_device(str(cfg.get("device", "auto")))

    run_dir = get_run_dir(
        experiments_dir=cfg.get("experiments_dir", "experiments"),
        project_name=cfg.get("project_name", "carracing-ppo"),
        run_name=cfg.get("run_name", ""),
    )

    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    plots_dir = os.path.join(run_dir, "plots")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save config + system info for reproducibility
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "system_info.txt"), "w", encoding="utf-8") as f:
        f.write(get_system_info() + "\n")

    env_id = cfg["env_name"]
    seed = int(cfg.get("seed", 42))

    # Build train/eval environments
    train_env = make_carracing_vec_env(
        env_id=env_id,
        n_envs=int(cfg.get("n_envs", 1)),
        seed=seed,
        grayscale=bool(cfg.get("grayscale", True)),
        resize=cfg.get("resize", 84),
        frame_stack=int(cfg.get("frame_stack", 4)),
        render_mode=None,
    )

    eval_env = make_carracing_vec_env(
        env_id=env_id,
        n_envs=1,
        seed=seed + 999,
        grayscale=bool(cfg.get("grayscale", True)),
        resize=cfg.get("resize", 84),
        frame_stack=int(cfg.get("frame_stack", 4)),
        render_mode=None,
    )

    # EvalCallback saves:
    # - best_model.zip into best_model_save_path
    # - evaluations.npz into log_path
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=int(cfg.get("eval_freq", 25000)),
        n_eval_episodes=int(cfg.get("n_eval_episodes", 20)),
        deterministic=bool(cfg.get("deterministic_eval", True)),
        render=False,
    )

    policy = str(cfg.get("policy", "CnnPolicy"))

    model = PPO(
        policy=policy,
        env=train_env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        n_steps=int(cfg.get("n_steps", 2048)),
        batch_size=int(cfg.get("batch_size", 64)),
        n_epochs=int(cfg.get("n_epochs", 10)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        target_kl=cfg.get("target_kl", None),
        verbose=int(cfg.get("verbose", 1)),
        device=device,
        tensorboard_log=(logs_dir if bool(cfg.get("tensorboard", True)) else None),
    )

    total_timesteps = int(cfg.get("total_timesteps", 3_000_000))
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # Save final model
    final_path = os.path.join(models_dir, "final_model.zip")
    model.save(final_path)

    # Evaluate final model quickly
    mean_r, std_r, _ = evaluate(
        model=model,
        env=eval_env,
        n_eval_episodes=int(cfg.get("n_eval_episodes", 20)),
        deterministic=bool(cfg.get("deterministic_eval", True)),
    )
    with open(os.path.join(run_dir, "final_eval.txt"), "w", encoding="utf-8") as f:
        f.write(f"Final Model - Mean reward: {mean_r:.2f} +/- {std_r:.2f}\n")

    # Plot evaluations (if available)
    eval_npz = os.path.join(logs_dir, "evaluations.npz")
    plot_path = os.path.join(plots_dir, "eval_curve.png")
    plot_evaluations_npz(eval_npz, plot_path, title=f"PPO Performance on {env_id}")

    # Close envs
    train_env.close()
    eval_env.close()

    return run_dir
