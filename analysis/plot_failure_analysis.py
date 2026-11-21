"""
Phân tích thất bại: Biểu đồ phân bố vị trí x cuối lượt
Phụ trách: SV3 (Giai đoạn 4: T3 1/12)

Đúng theo yêu cầu Mục 6.5 của đề bài:
"Tác tử thường chết ở đâu (biểu đồ vị trí x cuối lượt), vì sao, nhóm đã thử gì."
Đọc tệp JSON sinh ra bởi lệnh:
python -m mario_ppo.evaluate --ckpt <ckpt> --level <level> --out ket-qua.json
"""
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_failure_distribution(json_file: str, output_path: str = "analysis/failure_x_distribution.png"):
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"[!] Không tìm thấy tệp JSON kết quả đánh giá: {json_file}")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    x_positions = data.get("x_pos", [])
    level = data.get("level", "1-1")
    flag_rate = data.get("flag_rate", 0.0)
    mean_x = data.get("mean_x_pos", np.mean(x_positions) if x_positions else 0)

    if not x_positions:
        print(f"[!] Không có dữ liệu x_pos trong tệp {json_file}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Vẽ histogram phân bố vị trí x cuối lượt
    counts, bins, patches = plt.hist(x_positions, bins=15, color="#e74c3c", edgecolor="black", alpha=0.75)
    plt.axvline(mean_x, color="blue", linestyle="dashed", linewidth=2, label=f"Vị trí x trung bình: {mean_x:.0f}")
    plt.axvline(3160, color="green", linestyle=":", linewidth=2, label="Vị trí cột cờ (~3160)")

    plt.title(f"Phân tích thất bại: Phân bố vị trí x cuối lượt (Màn {level})\nFlag Rate: {flag_rate * 100:.1f}% ({len(x_positions)} lượt)", fontsize=12, fontweight="bold")
    plt.xlabel("Tọa độ x đạt được trước khi kết thúc lượt")
    plt.ylabel("Số lượt kết thúc tại tọa độ")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Đã lưu biểu đồ phân tích thất bại vào: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vẽ biểu đồ phân bố vị trí x cuối lượt")
    p.add_argument("--json", required=True, help="Đường dẫn tệp JSON do mario_ppo.evaluate sinh ra")
    p.add_argument("--out", default="analysis/failure_x_distribution.png", help="Đường dẫn tệp ảnh xuất ra")
    args = p.parse_args()
    plot_failure_distribution(args.json, args.out)
