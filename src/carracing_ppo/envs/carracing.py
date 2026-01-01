from __future__ import annotations

from typing import Optional

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


def _make_preprocess_wrapper(grayscale: bool, resize: Optional[int]):
    """
    Returns a callable that wraps a single Gymnasium env.

    We avoid Atari-specific wrappers here and use Gymnasium’s built-ins:
    - GrayScaleObservation(keep_dim=True) -> HxWx1
    - ResizeObservation -> (resize, resize)
    """
    def _wrap(env: gym.Env) -> gym.Env:
        if grayscale:
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        if resize is not None:
            env = gym.wrappers.ResizeObservation(env, (resize, resize))
        return env

    return _wrap


def make_carracing_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    grayscale: bool = True,
    resize: Optional[int] = 84,
    frame_stack: int = 4,
    render_mode: Optional[str] = None,
):
    """
    Creates a VecEnv suitable for SB3 CnnPolicy:
    - optional grayscale + resize
    - VecFrameStack (temporal context)
    - VecTransposeImage (channel-first for PyTorch CNN)
    """
    wrapper = _make_preprocess_wrapper(grayscale=grayscale, resize=resize)

    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    vec = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=wrapper,
        env_kwargs=env_kwargs if env_kwargs else None,
    )

    if frame_stack and frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=frame_stack)

    vec = VecTransposeImage(vec)
    return vec
