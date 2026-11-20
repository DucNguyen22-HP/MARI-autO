"""
Learning Curves Plotting Script
Responsible: SV3 (Giai đoạn 3 & 4: T7 28/11 đến T3 1/12)
Plots reward, approx KL, clip fraction, and entropy from progress.csv.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(log_file: str, output_path: str = "analysis/learning_curves.png"):
    if not os.path.exists(log_file):
        print(f"[!] Log file not found: {log_file}")
        return

    df = pd.read_csv(log_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward
    axes[0, 0].plot(df["step"], df["mean_reward"], color="#1f77b4", label="Mean Reward")
    axes[0, 0].set_title("Training Episode Reward")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Approx KL
    if "approx_kl" in df.columns:
        axes[0, 1].plot(df["step"], df["approx_kl"], color="#ff7f0e", label="Approx KL")
        axes[0, 1].axhline(y=0.03, color="red", linestyle=":", label="KL Warning Threshold (0.03)")
        axes[0, 1].set_title("Approximate KL Divergence")
        axes[0, 1].set_xlabel("Steps")
        axes[0, 1].set_ylabel("KL")
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Clip Fraction
    if "clip_fraction" in df.columns:
        axes[1, 0].plot(df["step"], df["clip_fraction"], color="#2ca02c", label="Clip Fraction")
        axes[1, 0].set_title("PPO Clip Fraction")
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Fraction")
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Policy Entropy
    if "entropy" in df.columns:
        axes[1, 1].plot(df["step"], df["entropy"], color="#9467bd", label="Policy Entropy")
        axes[1, 1].set_title("Policy Entropy")
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Saved learning curves plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, default="logs/progress.csv")
    parser.add_argument("--output", type=str, default="analysis/learning_curves.png")
    args = parser.parse_args()
    plot_learning_curves(args.log_file, args.output)
