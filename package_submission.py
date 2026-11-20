"""
Submission Packaging Script (Checklist Mục 7 của đề)
Responsible: SV3 (Điều phối đóng gói nhom-<số>.zip trước 3/12)

Checklist kiểm tra gói nộp:
[ ] Mã nguồn đầy đủ (src, configs, tests, scripts)
[ ] config.yaml, progress.csv, evals.csv của các lần chạy nộp
[ ] Checkpoint tốt nhất của 1-1 và của màn thưởng
[ ] Các tệp JSON và video do evaluate.py sinh ra
[ ] Báo cáo PDF (tối đa 8 trang)
[ ] Tệp trình chiếu
"""
import os
import zipfile
import argparse


def package_submission(group_id: str, output_dir: str = "."):
    zip_filename = f"nhom-{group_id}.zip"
    zip_path = os.path.join(output_dir, zip_filename)

    print(f"[*] Packaging submission for Group {group_id} -> {zip_path}")

    # Files and folders to include
    include_paths = [
        "src",
        "configs",
        "tests",
        "analysis",
        "train.py",
        "evaluate.py",
        "requirements.txt",
        "README.md",
        "logs",
        "eval_results",
        "checkpoints",
        "report",
        "slides",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if not os.path.exists(path):
                print(f"[!] Warning: Path not found: {path} (skipping)")
                continue

            if os.path.isfile(path):
                zipf.write(path, arcname=path)
                print(f"  + Added file: {path}")
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith((".pyc", ".pyo", ".DS_Store")) or "__pycache__" in root:
                            continue
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, start=".")
                        zipf.write(full_path, arcname=arcname)
                print(f"  + Added directory: {path}")

    print(f"\n[✓] Successfully packaged: {zip_path}")
    print("[*] Please verify against Checklist Section 7 before submitting!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package group submission")
    parser.add_argument("--group", type=str, default="01", help="Group number (e.g. 01)")
    args = parser.parse_args()
    package_submission(args.group)
