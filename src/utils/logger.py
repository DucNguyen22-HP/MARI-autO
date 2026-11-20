"""
Metrics Logging and Progress CSV Writer
Responsible: SV3 (Đánh giá, Điều phối, Báo cáo)
Produces: progress.csv, evals.csv (Required checklist in Section 7)
"""
import os
import csv
from typing import Dict, Any


class MetricsLogger:
    """
    Logs step metrics, loss values, rewards, and diagnostic signals (approx_kl, clip_frac, entropy).
    Outputs cleanly to progress.csv or evals.csv.
    """
    def __init__(self, log_dir: str, filename: str = "progress.csv"):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        self.headers = None
        self._file_initialized = os.path.exists(self.filepath)

    def log(self, metrics: Dict[str, Any]):
        if not self._file_initialized:
            self.headers = list(metrics.keys())
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
            self._file_initialized = True
        
        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers or list(metrics.keys()))
            writer.writerow(metrics)
