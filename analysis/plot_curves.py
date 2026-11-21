"""
Vẽ biểu đồ đường cong học tập từ progress.csv
Phụ trách: SV3 (Giai đoạn 3 & 4: T7 28/11 đến T3 1/12)

Đúng theo yêu cầu Mục 6.3 của đề bài:
"Theo số bước: lợi tức, vị trí x, tỉ lệ chạm cờ, entropy, approxKL, tỉ lệ cắt."
"""
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(log_file: str, output_path: str = "analysis/learning_curves.png"):
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"[!] Không tìm thấy tệp nhật ký: {log_file}")
        return

    df = pd.read_csv(log_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    step_col = "global_step" if "global_step" in df.columns else ("step" if "step" in df.columns else df.columns[0])
    steps = df[step_col]

    # Khởi tạo 6 đồ thị theo đúng Mục 6.3
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Lợi tức (ep_return)
    ret_col = "ep_return" if "ep_return" in df.columns else "mean_reward"
    if ret_col in df.columns:
        axes[0, 0].plot(steps, df[ret_col], color="#1f77b4", linewidth=1.5)
        axes[0, 0].set_title("1. Lợi tức trung bình (Episode Return)", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Lợi tức")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Vị trí x trung bình (x_pos)
    if "x_pos" in df.columns:
        axes[0, 1].plot(steps, df["x_pos"], color="#2ca02c", linewidth=1.5)
        axes[0, 1].axhline(y=3160, color="red", linestyle=":", label="Cột cờ (~3160)")
        axes[0, 1].set_title("2. Vị trí x trung bình cuối lượt", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Tọa độ x")
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Tỉ lệ chạm cờ (flag_rate)
    if "flag_rate" in df.columns:
        axes[1, 0].plot(steps, df["flag_rate"], color="#d62728", linewidth=1.5)
        axes[1, 0].axhline(y=0.8, color="green", linestyle="--", label="Ngưỡng đạt 80% (Mục 3.2)")
        axes[1, 0].set_title("3. Tỉ lệ chạm cờ (Flag Rate)", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Tỉ lệ (0.0 - 1.0)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Approx KL (approx_kl)
    if "approx_kl" in df.columns:
        axes[1, 1].plot(steps, df["approx_kl"], color="#ff7f0e", linewidth=1.2)
        axes[1, 1].axhline(y=0.03, color="red", linestyle=":", label="Cảnh báo bước quá lớn (0.03)")
        axes[1, 1].set_title("4. Xấp xỉ phân kỳ KL (Approx KL)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("KL")
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # 5. Tỉ lệ cắt (clipfrac)
    clip_col = "clipfrac" if "clipfrac" in df.columns else "clip_fraction"
    if clip_col in df.columns:
        axes[2, 0].plot(steps, df[clip_col], color="#9467bd", linewidth=1.2)
        axes[2, 0].set_title("5. Tỉ lệ mẫu bị cắt PPO (Clip Fraction)", fontsize=11, fontweight="bold")
        axes[2, 0].set_ylabel("Tỉ lệ cắt")
        axes[2, 0].set_xlabel("Số bước tác tử (Global Steps)")
        axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # 6. Entropy (entropy)
    if "entropy" in df.columns:
        axes[2, 1].plot(steps, df["entropy"], color="#8c564b", linewidth=1.2)
        axes[2, 1].set_title("6. Entropy của chính sách", fontsize=11, fontweight="bold")
        axes[2, 1].set_ylabel("Entropy")
        axes[2, 1].set_xlabel("Số bước tác tử (Global Steps)")
        axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Đã lưu 6 biểu đồ đường cong học tập vào: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vẽ đường cong học tập từ progress.csv")
    p.add_argument("--log", default="runs/1-1/sv1_run/progress.csv", help="Đường dẫn tệp progress.csv")
    p.add_argument("--out", default="analysis/learning_curves.png", help="Đường dẫn tệp ảnh xuất ra")
    args = p.parse_args()
    plot_learning_curves(args.log, args.out)
