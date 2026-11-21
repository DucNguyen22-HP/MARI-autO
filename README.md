# MARI-autO: Dạy Mario Vượt Màn Bằng PPO

> **Học phần:** Học Tăng Cường (Reinforcement Learning) - Học kỳ 1 năm học 2026–2027  
> **Nhóm thực hiện:** Nhóm 3 thành viên (`SV1`, `SV2`, `SV3`)  
> **Package chuẩn của môn học:** `mario_ppo`  
> **Môi trường:** `mo-supermario-v0` (MO-Gymnasium: phần thưởng véc-tơ 5 thành phần `[x_pos, time, death, coin, enemy]`)  
> **Quan sát:** 4 ảnh xám 84×84 xếp chồng (`channels = 4`, shape `(4, 84, 84)`)

---

## 📑 Mục lục
1. [Cấu trúc dự án chuẩn](#1-cấu-trúc-dự-án-chuẩn)
2. [Phân công công việc & Trách nhiệm thành viên](#2-phân-công-công-việc--trách-nhiệm-thành-viên)
3. [Lộ trình & Các mốc then chốt (Milestones)](#3-lộ-trình--các-mốc-then-chốt-milestones)
4. [Thang điểm & Tiêu chí đánh giá (Rubric)](#4-thang-điểm--tiêu-chí-đánh-giá-rubric)
5. [Cài đặt môi trường](#5-cài-đặt-môi-trường)
6. [Hướng dẫn chạy chương trình](#6-hướng-dẫn-chạy-chương-trình)
7. [Checklist gói nộp bài (Mục 7)](#7-checklist-gói-nộp-bài-mục-7)
8. [Quy định nghiêm ngặt từ đề bài](#8-quy-định-nghiêm-ngặt-từ-đề-bài)

---

## 1. Cấu trúc dự án chuẩn

Dự án tuân thủ tuyệt đối cấu trúc package `mario_ppo/` và bộ test `tests/` của đề bài để đảm bảo bot chấm điểm tự động Mục A (30 điểm cài đặt) không bị lỗi `ModuleNotFoundError`:

```text
MARI-autO/
├── .gitignore                      # Bỏ qua cache, runs nặng, videos, checkpoints
├── de-bai.md                       # Đề bài và quy định chi tiết từ giảng viên
├── pyproject.toml                  # Cấu hình cài đặt package mario-ppo & pytest
├── requirements.txt                # Danh sách thư viện phụ thuộc (NumPy < 2, Torch,...)
├── package_submission.py           # Script tự động kiểm tra và đóng gói nhom-<số>.zip (Mục 7)
├── README.md                       # Tài liệu hướng dẫn & phân công chi tiết
│
├── mario_ppo/                      # 📦 Package cốt lõi của môn học (KHÔNG ĐƯỢC ĐỔI TÊN)
│   ├── __init__.py
│   ├── wrappers.py                 # [SV2] FrameSkip, GrayResize, ScalarizeReward (CÓ SẴN: Fix lỗi NES)
│   ├── model.py                    # [SV3] ActorCritic (CNN 4 channels -> 84x84, policy, value)
│   ├── ppo.py                      # [SV1 & SV2] GAE, clipped_policy_loss, value_loss, ppo_loss, ppo_update
│   ├── env.py                      # [CÓ SẴN] Chuỗi wrapper tạo môi trường mo-supermario-v0
│   ├── evaluate.py                 # [CÓ SẴN - CẤM SỬA] Đánh giá chuẩn theo lệnh Mục 4
│   └── train.py                    # [CÓ SẴN] Vòng thu dữ liệu song song (AsyncVectorEnv) & cập nhật PPO
│
├── tests/                          # 🧪 Bộ kiểm thử tự động (30 điểm cài đặt - Mốc tối 20/11)
│   ├── conftest.py                 # Mock môi trường và cấu hình test
│   ├── test_wrappers.py            # [SV2] Kiểm tra FrameSkip, GrayResize, ScalarizeReward (6 điểm)
│   ├── test_model.py               # [SV3] Kiểm tra ActorCritic shape, logits, act (6 điểm)
│   ├── test_gae.py                 # [SV2] Kiểm tra compute_gae & xử lý kết thúc lượt (9 điểm)
│   ├── test_ppo.py                 # [SV1] Kiểm tra clipped_loss, value_loss, ppo_loss, update (9 điểm)
│   └── test_pipeline.py            # [SV3 & Cả nhóm] Kiểm tra tích hợp toàn diện vòng huấn luyện
│
├── configs/                        # ⚙️ Cấu hình huấn luyện OmegaConf
│   ├── default.yaml                # Cấu hình mặc định gốc từ giảng viên
│   ├── sv1_stage1_1.yaml           # [SV1] Cấu hình chạy máy 1 màn 1-1
│   ├── sv1_bonus_transfer.yaml     # [SV1] Cấu hình học chuyển giao màn thưởng (1-2/3-1)
│   ├── sv2_gamma_ablation.yaml     # [SV2] Cấu hình thí nghiệm so sánh (γ = 0.99 vs 0.90)
│   └── sv3_seed2_baseline.yaml     # [SV3] Cấu hình chạy đối chứng Seed thứ hai
│
├── analysis/                       # 📊 [SV3] Trực quan hóa dữ liệu phục vụ Báo cáo
│   ├── plot_curves.py              # Vẽ 6 đồ thị theo Mục 6.3 từ progress.csv
│   └── plot_failure_analysis.py    # Vẽ biểu đồ phân bố vị trí x cuối lượt từ JSON của evaluate.py
│
├── checkpoints/                    # 💾 Lưu trữ checkpoint tốt nhất
│   ├── README.md                   # Hướng dẫn lưu best_1-1.pt và best_bonus.pt
│   └── .gitkeep
│
├── runs/                           # 📈 Thư mục sinh ra tự động khi train (config, progress.csv, evals.csv, ckpt, video)
├── report/                         # 📝 Báo cáo đồ án (Tối đa 8 trang PDF)
└── slides/                         # 📽️ Slide trình chiếu thuyết trình (10 phút)
```

---

## 2. Phân công công việc & Trách nhiệm thành viên

Nhóm phân chia ranh giới code rõ ràng theo đúng lịch trình và rubric:

| Thành viên | Nhiệm vụ Cài đặt (17/11 - 20/11)<br>*(Chỉ cài các hàm TODO)* | Nhiệm vụ Huấn luyện & Thí nghiệm (21/11 - 30/11) | Báo cáo & Nộp bài (01/12 - 03/12) |
| :--- | :--- | :--- | :--- |
| **SV1** | **Thuật toán PPO (`mario_ppo/ppo.py`)**<br>• `clipped_policy_loss`<br>• `value_loss`<br>• `approx_kl_and_clipfrac`<br>• `ppo_loss`, `ppo_update`<br>• Test: `test_ppo.py` (9 điểm) | • **Máy 1**: Huấn luyện màn 1-1 riêng<br>• Huấn luyện màn thưởng (ưu tiên 1-2 hoặc 3-1) bằng Transfer Learning từ checkpoint 1-1 | • Viết phần thiết lập huấn luyện & lý thuyết PPO<br>• Viết nội dung thí nghiệm so sánh<br>• Rà soát gói nộp cuối cùng |
| **SV2** | **GAE & Wrappers (`mario_ppo/`)**<br>• `ppo.py`: `compute_gae`<br>• `wrappers.py`: `FrameSkip`, `GrayResize`, `ScalarizeReward`<br>• Test: `test_gae.py` (9 điểm), `test_wrappers.py` (6 điểm) | • **Máy 2**: Huấn luyện màn 1-1 riêng<br>• Chạy thí nghiệm so sánh Ablation trên 1-1 (ví dụ $\gamma = 0.99$ so với baseline $0.90$) | • Viết phần chi tiết cài đặt & bảng kết quả pytest<br>• Viết mục định hình phần thưởng (reward shaping)<br>• Dựng Slide (cài đặt & thí nghiệm) |
| **SV3** | **Kiến trúc Mạng & Đánh giá (`mario_ppo/`)**<br>• `model.py`: `ActorCritic.__init__`, `forward`<br>• Giúp SV2 viết `GrayResize` / `FrameSkip`<br>• Điều phối `tests/test_pipeline.py`<br>• Test: `test_model.py` (6 điểm) | • **Máy 3**: Huấn luyện màn 1-1 riêng<br>• Chạy đối chứng Seed thứ hai với cấu hình gốc<br>• Dùng `evaluate.py` kiểm tra checkpoint trung gian<br>• Vẽ đường cong học & biểu đồ thất bại | • Viết mục phân tích đường cong học & phân tích thất bại (x_pos)<br>• Dựng Slide (kết quả), ghép PDF báo cáo<br>• Điều phối đóng gói `nhom-<số>.zip` |

---

## 3. Lộ trình & Các mốc then chốt (Milestones)

- **Mốc 1 (Tối 20/11 - Bắt buộc):** Mọi bài kiểm tra trong `tests/` phải qua `pytest` (Mục A: 30 điểm). Nếu còn lỗi, sửa dứt điểm trong sáng 21/11 rồi mới khởi động máy chạy.
- **Mốc 2 (25/11):** Chốt checkpoint 1-1 đạt `flag_rate >= 0.8` (Mục B: 25 điểm). Chốt màn thưởng (ưu tiên 1-2 hoặc 3-1).
- **Mốc 3 (30/11):** Chốt checkpoint màn thưởng và dữ liệu thí nghiệm so sánh (Mục C: 15 điểm). Chạy `evaluate.py` xuất file JSON và video.
- **Mốc 4 (03/12 - Hạn nội bộ nhóm):** Hoàn tất báo cáo PDF (tối đa 8 trang), slide thuyết trình và đóng gói tệp `nhom-<số>.zip` (sớm 6 ngày trước hạn nộp chính thức 09/12).
- **Vùng dự phòng (04/12 - 09/12):** Huấn luyện bổ sung nếu màn thưởng chưa đạt cờ, tập thuyết trình thử 10 phút và trả lời hỏi đáp 5 phút.

---

## 4. Thang điểm & Tiêu chí đánh giá (Rubric)

| Hạng mục | Điểm | Cách tính theo Mục 5 đề bài |
| :--- | :---: | :--- |
| **A. Cài đặt** | **30** | Tỉ lệ bài kiểm tra qua: `wrappers` (6đ), `model` (6đ), `GAE` (9đ), `PPO` (9đ) |
| **B. Màn 1-1** | **25** | $25 \cdot \min(1, \frac{f}{0.8})$ với $f$ là `flag_rate` theo lệnh đánh giá Mục 4 |
| **C. Màn thưởng** | **15** | $15 \cdot \min(1, \frac{f}{0.8})$ trên màn thưởng đã chọn trong danh sách |
| **D. Báo cáo** | **25** | Tối đa 8 trang PDF: Cài đặt, Cấu hình, Đường cong học, Thí nghiệm so sánh, Phân tích lỗi, Phân công |
| **E. Trình bày & Demo** | **5** | 10 phút trình bày slide và video demo, 5 phút hỏi đáp |
| **Điểm thưởng** | *(tối đa +10)* | Vượt thêm một màn ngoài danh sách (ví dụ 4-2, 7-1, 8-1) |

---

## 5. Cài đặt môi trường

> [!IMPORTANT]
> Bắt buộc dùng **Python 3.10** và **NumPy < 2.0.0** vì `nes-py 8.2.1` bị tràn số trên NumPy 2.x.

```bash
# 1. Tạo môi trường ảo với Python 3.10
conda create -n mario-ppo python=3.10 -y
conda activate mario-ppo

# 2. Cài đặt package mario_ppo ở chế độ editable
pip install -e .

# Hoặc cài đặt qua requirements.txt:
pip install -r requirements.txt
```

---

## 6. Hướng dẫn chạy chương trình

### 6.1. Chạy bộ kiểm thử tự động (Mục A - 30 điểm)
```bash
# Chạy toàn bộ bài test:
pytest -v

# Chạy riêng từng bài test của từng người:
pytest tests/test_wrappers.py -v   # SV2 (6 điểm)
pytest tests/test_model.py -v      # SV3 (6 điểm)
pytest tests/test_gae.py -v        # SV2 (9 điểm)
pytest tests/test_ppo.py -v        # SV1 (9 điểm)
pytest tests/test_pipeline.py -v   # Tích hợp toàn diện
```

### 6.2. Huấn luyện (Giai đoạn 2 & 3)
```bash
# Huấn luyện theo cấu hình mặc định:
python -m mario_ppo.train

# Ghi đè tham số trực tiếp trên dòng lệnh:
python -m mario_ppo.train level=1-4 algo.total_steps=8000000

# Huấn luyện theo cấu hình riêng của từng thành viên:
python -m mario_ppo.train --config configs/sv1_stage1_1.yaml        # Máy 1 (SV1)
python -m mario_ppo.train --config configs/sv2_gamma_ablation.yaml  # Máy 2 (SV2 - Ablation gamma)
python -m mario_ppo.train --config configs/sv3_seed2_baseline.yaml  # Máy 3 (SV3 - Seed 2)

# Chạy tiếp khi bị ngắt (resume):
python -m mario_ppo.train resume=runs/1-1/<run>/checkpoints/step_002000000.pt

# Học chuyển giao sang màn thưởng (khởi tạo từ checkpoint 1-1):
python -m mario_ppo.train --config configs/sv1_bonus_transfer.yaml resume=checkpoints/best_1-1.pt
```

> [!TIP]
> **Chẩn đoán ở mốc 500.000 bước (CN 22/11):**  
> `train.py` tự động in ra màn hình các chỉ số sau mỗi lần cập nhật:  
> `sps`: Tốc độ (bước/giây) | `ep_return`: Lợi tức | `x`: Vị trí x | `flag`: Tỉ lệ chạm cờ | `kl`: approxKL.  
> *Nếu `approx_kl > 0.03` liên tục hoặc Mario không tăng được vị trí x, cần dừng sớm để chỉnh siêu tham số/định hình phần thưởng.*

### 6.3. Đánh giá mô hình (Lệnh chuẩn Mục 4 của giảng viên - KHÔNG ĐƯỢC ĐỔI CÚ PHÁP)
```bash
# Đánh giá checkpoint màn 1-1:
python -m mario_ppo.evaluate --ckpt checkpoints/best_1-1.pt --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4

# Đánh giá checkpoint màn thưởng (ví dụ 1-2):
python -m mario_ppo.evaluate --ckpt checkpoints/best_bonus.pt --level 1-2 --episodes 20 --seed 2026 --out ket-qua-1-2.json --video 1-2.mp4
```

### 6.4. Trực quan hóa dữ liệu phục vụ Báo cáo
```bash
# 1. Vẽ 6 đường cong học tập theo Mục 6.3:
python analysis/plot_curves.py --log runs/1-1/sv1_run/progress.csv --out analysis/learning_curves.png

# 2. Vẽ biểu đồ phân tích thất bại (vị trí x cuối lượt) theo Mục 6.5:
python analysis/plot_failure_analysis.py --json ket-qua-1-1.json --out analysis/failure_x_distribution.png
```

### 6.5. Đóng gói nộp bài (Mục 7)
```bash
python package_submission.py --group 01
```

---

## 7. Checklist gói nộp bài (Mục 7)

Tệp nén `nhom-<số>.zip` nộp bài phải đảm bảo đầy đủ:
- [x] **Mã nguồn:** Toàn bộ package `mario_ppo/`, `tests/`, `configs/`, `pyproject.toml`, `requirements.txt`.
- [x] **Cấu hình & Nhật ký:** `config.yaml`, `progress.csv`, `evals.csv` của các lần chạy nộp.
- [x] **Checkpoints tốt nhất:** Checkpoint tốt nhất của màn 1-1 (`checkpoints/best_1-1.pt`) và của màn thưởng (`best_bonus.pt`).
- [x] **Kết quả đánh giá Mục 4:** Các tệp JSON (`ket-qua-1-1.json`, ...) và video MP4 (`1-1.mp4`, ...) do `evaluate.py` sinh ra.
- [x] **Báo cáo PDF:** Tối đa 8 trang trong thư mục `report/`.
- [x] **Slide thuyết trình:** Tệp trình chiếu trong thư mục `slides/`.

---

## 8. Quy định nghiêm ngặt từ đề bài

1. **Cấm sửa `mario_ppo/evaluate.py`:** Giảng viên sẽ chạy lại đúng lệnh ở Mục 4 bằng script gốc trên checkpoint nộp bài.
2. **Cấm sửa phần `eval_mode` trong `mario_ppo/env.py`:** Mẹo huấn luyện bổ sung chỉ được đặt trong khối `if not eval_mode:` của `make_env`.
3. **Đầu vào của mạng:** Chỉ gồm 4 khung hình xám đã xếp chồng (shape `(4, 84, 84)`); tuyệt đối không đọc RAM trò chơi hay tọa độ trong `info` làm đầu vào mạng.
4. **Không dùng checkpoint bên ngoài:** Toàn bộ checkpoint phải được tự huấn luyện và tái tạo được từ cấu hình đã nộp.
5. **Git Commit cá nhân:** Mỗi thành viên phải commit riêng phần việc của mình trên Git để lịch sử commit khớp chính xác với bảng phân công trong báo cáo.
