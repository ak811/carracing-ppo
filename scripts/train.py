from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carracing_ppo.training.train_sb3 import train
from carracing_ppo.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v3 (SB3)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = train(cfg)
    print(f"\nRun complete. Outputs in: {run_dir}\n")


if __name__ == "__main__":
    main()
