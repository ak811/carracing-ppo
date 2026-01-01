import numpy as np

from carracing_ppo.envs.carracing import make_carracing_vec_env


def test_env_build_and_step():
    env = make_carracing_vec_env(
        env_id="CarRacing-v3",
        n_envs=1,
        seed=0,
        grayscale=True,
        resize=84,
        frame_stack=4,
        render_mode=None,
    )
    obs = env.reset()
    assert obs is not None

    action = np.zeros((1, 3), dtype=np.float32)  # steer, gas, brake
    obs, rewards, dones, infos = env.step(action)
    assert obs is not None
    env.close()
