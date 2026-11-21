"""
Script đóng gói bài nộp: nhom-<số>.zip
Phụ trách: SV3 (Điều phối đóng gói trước hạn nội bộ 3/12)

Đúng theo Checklist Mục 7 của đề bài:
[ ] Mã nguồn đầy đủ (mario_ppo/, tests/, configs/, pyproject.toml, requirements.txt, ...)
[ ] config.yaml, progress.csv, evals.csv của các lần chạy nộp
[ ] Checkpoint tốt nhất của 1-1 và của màn thưởng
[ ] Các tệp JSON và video do evaluate.py sinh ra theo Mục 4
[ ] Báo cáo PDF (tối đa 8 trang)
[ ] Tệp trình chiếu
"""
import argparse
import os
from pathlib import Path
import zipfile


REQUIRED_CHECKLIST = [
    ("Mã nguồn mario_ppo/", "mario_ppo"),
    ("Bộ kiểm thử tests/", "tests"),
    ("Cấu hình configs/", "configs"),
    ("Tệp pyproject.toml", "pyproject.toml"),
    ("Tệp README.md", "README.md"),
]


def check_and_package(group_num: str, output_dir: str = "."):
    group_str = str(group_num).strip().zfill(2)
    zip_name = f"nhom-{group_str}.zip"
    zip_path = Path(output_dir) / zip_name

    print("=" * 60)
    print(f"[*] KIỂM TRA & ĐÓNG GÓI BÀI NỘP: {zip_name}")
    print("=" * 60)

    # 1. Kiểm tra các thành phần bắt buộc
    missing_items = []
    for desc, rel_path in REQUIRED_CHECKLIST:
        p = Path(rel_path)
        if p.exists():
            print(f" [✓] {desc} -> OK")
        else:
            print(f" [✗] {desc} -> THIẾU ({rel_path})")
            missing_items.append(desc)

    # 2. Thu thập tệp nộp
    items_to_zip = [
        "mario_ppo",
        "tests",
        "configs",
        "analysis",
        "checkpoints",
        "runs",
        "report",
        "slides",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "package_submission.py",
    ]

    print("\n[*] Tiến hành nén tệp...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items_to_zip:
            p = Path(item)
            if not p.exists():
                continue
            if p.is_file():
                zf.write(p, arcname=str(p))
                print(f"  + Thêm file: {p}")
            elif p.is_dir():
                for root, _, files in os.walk(p):
                    for file in files:
                        if file.endswith((".pyc", ".pyo", ".DS_Store")) or "__pycache__" in root:
                            continue
                        f_path = Path(root) / file
                        zf.write(f_path, arcname=str(f_path))
                print(f"  + Thêm thư mục: {p}/")

        # Thêm các tệp JSON và MP4 đánh giá ở thư mục gốc (nếu có)
        for ext in ("*.json", "*.mp4"):
            for f in Path(".").glob(ext):
                zf.write(f, arcname=f.name)
                print(f"  + Thêm kết quả đánh giá: {f.name}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"[✓] Đã tạo gói nộp thành công: {zip_path} ({size_mb:.2f} MB)")
    print("=" * 60)
    print("Nhóm hãy kiểm tra lại lần cuối trước khi nộp:")
    print("  [ ] Đã có checkpoint tốt nhất của 1-1 và màn thưởng chưa?")
    print("  [ ] Đã có các tệp JSON và MP4 theo đúng lệnh Mục 4 chưa?")
    print("  [ ] Đã có báo cáo PDF (tối đa 8 trang) trong report/ chưa?")
    print("  [ ] Đã có slide thuyết trình trong slides/ chưa?")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Đóng gói bài nộp nhom-<số>.zip")
    p.add_argument("--group", default="01", help="Số thứ tự nhóm (ví dụ: 01, 02)")
    args = p.parse_args()
    check_and_package(args.group)
