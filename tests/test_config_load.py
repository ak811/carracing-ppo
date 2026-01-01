from carracing_ppo.utils.io import load_config


def test_load_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("env_name: CarRacing-v3\nseed: 1\n", encoding="utf-8")
    cfg = load_config(str(cfg_path))
    assert cfg["env_name"] == "CarRacing-v3"
    assert cfg["seed"] == 1
