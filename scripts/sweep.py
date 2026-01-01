from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carracing_ppo.training.train_sb3 import train
from carracing_ppo.utils.io import load_config, dump_config


def dict_product(grid: dict):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def main():
    parser = argparse.ArgumentParser(description="Grid sweep runner for CarRacing PPO (SB3).")
    parser.add_argument("--sweep", required=True, help="Path to sweep.yaml")
    args = parser.parse_args()

    sweep_cfg = load_config(args.sweep)
    base_cfg = load_config(sweep_cfg["base_config"])
    grid = sweep_cfg.get("grid", {})
    overrides = sweep_cfg.get("overrides", {})

    for i, params in enumerate(dict_product(grid), start=1):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        cfg.update(params)

        cfg["run_name"] = f"sweep_{i}_" + "_".join([f"{k}={params[k]}" for k in sorted(params.keys())])

        run_dir = train(cfg)
        dump_config(params, os.path.join(run_dir, "sweep_params.yaml"))
        print(f"Sweep run {i} done -> {run_dir}")


if __name__ == "__main__":
    main()
