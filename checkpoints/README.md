# Checkpoints Directory

Thư mục lưu trữ các checkpoint trong quá trình huấn luyện:
- `latest.pt`: Checkpoint gần nhất được cập nhật tự động qua các rollout.
- `best_1-1.pt`: Checkpoint đạt hiệu suất tốt nhất trên màn 1-1 (Mục 7 yêu cầu).
- `best_bonus.pt`: Checkpoint tốt nhất trên màn thưởng (Mục 7 yêu cầu).

*Lưu ý: Không commit các file checkpoint dung lượng lớn lên git. Đóng gói vào tệp zip nộp bài bằng lệnh `python package_submission.py`.*
