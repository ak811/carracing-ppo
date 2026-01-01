from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from carracing_ppo.utils.plotting import plot_evaluations_npz


def main():
    parser = argparse.ArgumentParser(description="Plot SB3 EvalCallback evaluations.npz")
    parser.add_argument("--npz", required=True, help="Path to evaluations.npz")
    parser.add_argument("--out", required=True, help="Output png path")
    parser.add_argument("--title", default="PPO Evaluation Curve")
    args = parser.parse_args()

    ok = plot_evaluations_npz(args.npz, args.out, title=args.title)
    print("Saved plot." if ok else "Could not plot (file missing or invalid).")


if __name__ == "__main__":
    main()
