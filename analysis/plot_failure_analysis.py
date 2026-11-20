"""
Failure Analysis & Final X-Position Distribution Plotting Script
Responsible: SV3 (Giai đoạn 4: T3 1/12)
Analyzes where Mario gets stuck or dies across evaluation episodes.
"""
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_failure_distribution(eval_json: str, output_path: str = "analysis/failure_x_distribution.png"):
    if not os.path.exists(eval_json):
        print(f"[!] Evaluation JSON not found: {eval_json}")
        return

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    x_positions = [ep["max_x"] for ep in episodes]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.hist(x_positions, bins=15, color="#e74c3c", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(x_positions), color="blue", linestyle="dashed", linewidth=2, label=f"Mean X: {np.mean(x_positions):.1f}")
    plt.axvline(3160, color="green", linestyle=":", linewidth=2, label="Flag Pole (~3160)")

    plt.title(f"Final X-Position Distribution ({data.get('env_id', 'Mario')})")
    plt.xlabel("Mario Max X-Coordinate Reached")
    plt.ylabel("Episode Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Saved failure distribution plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=str, required=True, help="Path to evaluation JSON result")
    parser.add_argument("--output", type=str, default="analysis/failure_x_distribution.png")
    args = parser.parse_args()
    plot_failure_distribution(args.eval_json, args.output)
