# MARI-autO: Dạy Mario Vượt Màn Bằng PPO

> **Đồ án môn học:** Học Tăng Cường (Reinforcement Learning)  
> **Nhóm thực hiện:** 3 thành viên (`SV1`, `SV2`, `SV3`)  
> **Thuật toán chính:** Proximal Policy Optimization (PPO) kết hợp Generalized Advantage Estimation (GAE)  
> **Môi trường:** `gym-super-mario-bros` (NES)

---

## 📑 Mục lục
1. [Cấu trúc dự án](#1-cấu-trúc-dự-án)
2. [Phân công công việc & Trách nhiệm thành viên](#2-phân-công-công-việc--trách-nhiệm-thành-viên)
3. [Lộ trình & Các mốc then chốt (Milestones)](#3-lộ-trình--các-mốc-then-chốt-milestones)
4. [Checklist gói nộp bài (Mục 7)](#4-checklist-gói-nộp-bài-mục-7)
5. [Cài đặt môi trường](#5-cài-đặt-môi-trường)
6. [Hướng dẫn chạy chương trình](#6-hướng-dẫn-chạy-chương-trình)
7. [Chiến lược quản trị rủi ro & Git Workflow](#7-chiến-lược-quản-trị-rủi-ro--git-workflow)

---

## 1. Cấu trúc dự án

Dự án được tổ chức theo module hóa cao, phân định ranh giới code rõ ràng tương ứng với nhiệm vụ của từng thành viên:

```text
MARI-autO/
├── .gitignore                      # Cấu hình bỏ qua tệp nhị phân, cache, checkpoint nặng
├── README.md                       # Tài liệu hướng dẫn & cấu trúc tổng thể dự án
├── requirements.txt                # Danh sách thư viện phụ thuộc (PyTorch, Gym, OpenCV,...)
├── train.py                        # Script huấn luyện chính (chẩn đoán 500k bước, transfer learning)
├── evaluate.py                     # Script đánh giá checkpoint (chuẩn Mục 4: tính flag_rate, xuất JSON/CSV, video)
├── package_submission.py           # Script tự động kiểm tra và đóng gói nhom-<số>.zip
│
├── configs/                        # Thư mục lưu trữ cấu hình thí nghiệm (YAML)
│   ├── base_config.yaml            # Cấu hình mặc định dùng chung
│   ├── sv1_stage1_1.yaml           # [SV1] Cấu hình huấn luyện Stage 1-1
│   ├── sv1_bonus_transfer.yaml     # [SV1] Cấu hình học chuyển giao màn thưởng (1-2/3-1)
│   ├── sv2_gamma_ablation.yaml     # [SV2] Cấu hình thí nghiệm so sánh (γ = 0.9 vs 0.99)
│   └── sv3_seed2_baseline.yaml     # [SV3] Cấu hình chạy đối chứng với seed thứ hai
│
├── src/                            # Mã nguồn lõi thuật toán
│   ├── __init__.py
│   ├── algo/                       # [SV1] Phụ trách Thuật toán PPO
│   │   ├── __init__.py
│   │   └── ppo.py                  # clipped_policy_loss, value_loss, approx_kl_and_clipfrac, ppo_loss, ppo_update
│   ├── env/                        # [SV2] Phụ trách Môi trường & Tiền xử lý
│   │   ├── __init__.py
│   │   ├── wrappers.py             # FrameSkip, GrayResize, ScalarizeReward (định hình phần thưởng)
│   │   └── make_env.py             # Hàm khởi tạo môi trường Mario & thiết lập Joypad action space
│   ├── models/                     # [SV3] Phụ trách Kiến trúc Mạng Nơ-ron
│   │   ├── __init__.py
│   │   └── actor_critic.py         # Mạng ActorCritic (CNN Feature Extractor + Policy Head + Value Head)
│   └── utils/                      # [SV2 & SV3] Các tiện ích hỗ trợ
│       ├── __init__.py
│       ├── gae.py                  # [SV2] compute_gae (Generalized Advantage Estimation)
│       ├── buffer.py               # [SV2/SV3] RolloutBuffer lưu trữ quỹ đạo on-policy
│       ├── logger.py               # [SV3] Logger ghi progress.csv và evals.csv
│       └── seed.py                 # [SV3] Hàm set_seed đảm bảo tính tái lập thí nghiệm
│
├── tests/                          # [Cả nhóm] Kiểm thử tự động (Pytest) - Mốc tối 20/11
│   ├── __init__.py
│   ├── conftest.py                 # Mock environment và fixtures kiểm thử nhanh
│   ├── test_ppo.py                 # [SV1] Unit test các hàm loss và update của PPO
│   ├── test_gae.py                 # [SV2] Unit test thuật toán GAE & xử lý trạng thái done
│   ├── test_env.py                 # [SV2] Unit test các wrapper FrameSkip, GrayResize, Reward
│   ├── test_model.py               # [SV3] Unit test kích thước tensor và phân phối hành động
│   └── test_pipeline.py            # [SV3 & Cả nhóm] Integration test luồng huấn luyện tích hợp
│
├── analysis/                       # [SV3] Trực quan hóa & Phân tích dữ liệu thí nghiệm
│   ├── plot_curves.py              # Vẽ biểu đồ đường cong học (Reward, KL, ClipFrac, Entropy)
│   └── plot_failure_analysis.py    # Phân tích vị trí Mario chết/vướng (biểu đồ phân bố max x_pos)
│
├── checkpoints/                    # Lưu trữ trọng số mô hình đã huấn luyện
│   ├── .gitkeep
│   ├── README.md                   # Hướng dẫn lưu checkpoint tốt nhất
│   ├── best_1-1.pt                 # [Checklist] Checkpoint tốt nhất trên màn 1-1
│   └── best_bonus.pt               # [Checklist] Checkpoint tốt nhất trên màn thưởng
│
├── logs/                           # Nhật ký huấn luyện và file kết quả
│   ├── .gitkeep
│   └── progress.csv                # [Checklist] Tiến độ huấn luyện (bước, reward, approx_kl, sps)
│
├── eval_results/                   # Kết quả đánh giá mô hình
│   ├── .gitkeep
│   ├── evals.csv                   # [Checklist] Bảng tổng hợp các lần đánh giá
│   └── *.json & *.mp4              # [Checklist] File JSON chi tiết và video minh họa
│
├── report/                         # Báo cáo đồ án (Tối đa 8 trang PDF)
│   └── README.md                   # Phân công viết từng phần và khung sườn báo cáo
│
└── slides/                         # Slide thuyết trình (10 phút)
    └── README.md                   # Phân công dựng slide và chuẩn bị demo
```

---

## 2. Phân công công việc & Trách nhiệm thành viên

| Thành viên | Trọng tâm phụ trách | Nhiệm vụ kỹ thuật chính (Code & Tests) | Nhiệm vụ Thí nghiệm (21/11 - 30/11) | Báo cáo & Nộp bài (01/12 - 03/12) |
| :--- | :--- | :--- | :--- | :--- |
| **SV1** | **Thuật toán PPO & Học chuyển giao** | • Cài đặt `clipped_policy_loss`, `value_loss`<br>• Cài đặt `approx_kl_and_clipfrac`, `ppo_loss`, `ppo_update`<br>• Viết `tests/test_ppo.py` | • **Máy 1**: Huấn luyện màn 1-1 cấu hình riêng<br>• Huấn luyện màn thưởng (1-2/3-1) chuyển giao từ 1-1 | • Viết phần thiết lập huấn luyện & lý thuyết PPO<br>• Viết phân tích thí nghiệm so sánh<br>• Kiểm tra gói nộp cuối |
| **SV2** | **GAE, Môi trường & Định hình thưởng** | • Cài đặt `compute_gae`<br>• Cài đặt `FrameSkip`, `GrayResize`, `ScalarizeReward`<br>• Viết `tests/test_gae.py`, `tests/test_env.py` | • **Máy 2**: Huấn luyện màn 1-1 cấu hình riêng<br>• Thí nghiệm so sánh (ablation: $\gamma=0.9$ vs $0.99$) | • Viết chi tiết cài đặt & bảng kết quả pytest<br>• Viết mục định hình phần thưởng<br>• Dựng Slide (cài đặt & thí nghiệm) |
| **SV3** | **Kiến trúc Mạng, Đánh giá & Điều phối** | • Cài đặt mạng `ActorCritic` (CNN + Heads)<br>• Cài đặt `evaluate.py`, `logger.py`, `seed.py`<br>• Viết `tests/test_model.py`, `tests/test_pipeline.py`<br>• Hỗ trợ viết wrappers | • **Máy 3**: Huấn luyện màn 1-1 cấu hình riêng<br>• Chạy đối chứng Seed thứ 2 với cấu hình gốc<br>• Vẽ biểu đồ học tập & phân tích lỗi (x_pos) | • Tổng hợp đường cong học & phân tích thất bại<br>• Dựng Slide (kết quả), ghép PDF báo cáo<br>• Điều phối đóng gói `nhom-<số>.zip` |

---

## 3. Lộ trình & Các mốc then chốt (Milestones)

```text
17/11           20/11            25/11            30/11             03/12         09/12
  |---------------|----------------|----------------|-----------------|-------------|
     Giai đoạn 1       Giai đoạn 2      Giai đoạn 3       Giai đoạn 4     Dự phòng & Nộp
    (Cài đặt code    (Huấn luyện 1-1  (Màn thưởng &    (Báo cáo, Slide   (Tập thuyết trình,
     & Pytest)        & Chốt ckpt)     Thí nghiệm)      & Đóng gói ZIP)   Nộp chính thức)
```

- **Mốc 1 (Tối 20/11 - Bắt buộc):** Tất cả các test trong `tests/` phải `PASS` qua `pytest`. Nếu có lỗi, xử lý dứt điểm trước trưa 21/11.
- **Mốc 2 (25/11):** Chốt checkpoint tốt nhất màn 1-1 (`checkpoints/best_1-1.pt`) và chốt màn thưởng (ưu tiên 1-2 hoặc 3-1).
- **Mốc 3 (30/11):** Chốt checkpoint màn thưởng (`best_bonus.pt`) và dữ liệu thí nghiệm so sánh (gamma, seed 2).
- **Mốc 4 (03/12 - Hạn nội bộ nhóm):** Hoàn tất báo cáo PDF (tối đa 8 trang), slide thuyết trình và tệp nộp bài `nhom-<số>.zip` (sớm 6 ngày trước hạn 9/12).
- **Vùng dự phòng (04/12 - 09/12):** Huấn luyện bổ sung nếu màn thưởng chưa đạt `flag_rate >= 0.8`, tập thuyết trình 10 phút và giải đáp phản biện.

---

## 4. Checklist gói nộp bài (Mục 7)

Trước khi nộp bài, kiểm tra đầy đủ các thành phần thông qua script đóng gói:
- [x] **Mã nguồn đầy đủ:** Thư mục `src/`, `configs/`, `tests/`, `analysis/`, các script `train.py`, `evaluate.py`.
- [x] **Tệp cấu hình & Dữ liệu quá trình:** `config.yaml`, `logs/progress.csv`, `eval_results/evals.csv`.
- [x] **Checkpoints tốt nhất:** `checkpoints/best_1-1.pt` và `checkpoints/best_bonus.pt`.
- [x] **Kết quả đánh giá:** Các file JSON chi tiết và video `.mp4` được sinh ra từ `evaluate.py`.
- [x] **Báo cáo PDF:** `report/baocao.pdf` (tối đa 8 trang theo quy định).
- [x] **Tệp trình chiếu:** `slides/presentation.pptx` (hoặc PDF).

---

## 5. Cài đặt môi trường

```bash
# 1. Tạo và kích hoạt môi trường ảo
conda create -n mario-ppo python=3.10 -y
conda activate mario-ppo

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 6. Hướng dẫn chạy chương trình

### 6.1. Chạy kiểm thử tự động (Pytest) - Mốc 20/11
Chạy kiểm thử toàn bộ hệ thống trước khi bắt đầu huấn luyện:
```bash
# Kiểm thử toàn bộ
pytest -v

# Hoặc kiểm thử riêng integration pipeline:
pytest tests/test_pipeline.py -v -s
```

### 6.2. Huấn luyện màn 1-1 (Giai đoạn 2)
Mỗi thành viên chạy cấu hình được phân công trên máy của mình:
```bash
# SV1:
python train.py --config configs/sv1_stage1_1.yaml

# SV2 (Thí nghiệm Ablation Gamma = 0.9):
python train.py --config configs/sv2_gamma_ablation.yaml

# SV3 (Chạy Baseline với Seed 2):
python train.py --config configs/sv3_seed2_baseline.yaml
```

> **Lưu ý chẩn đoán ở 500.000 bước (CN 22/11):**  
> Kiểm tra log trên màn hình:
> - `Mean Reward`: Có xu hướng tăng dần theo thời gian.
> - `Approx KL`: Bình thường trong khoảng `0.005 - 0.02`. Nếu `> 0.03`, chính sách cập nhật quá mạnh, cần giảm learning rate.
> - `Clip Fraction`: Thường dao động trong khoảng `0.1 - 0.25`.
> - `Entropy`: Giảm từ từ, tránh hiện tượng sụp đổ sớm về 0 (policy collapse).

### 6.3. Huấn luyện màn thưởng bằng Transfer Learning (Giai đoạn 3)
Khởi động từ checkpoint tốt nhất của màn 1-1:
```bash
python train.py --config configs/sv1_bonus_transfer.yaml --pretrained checkpoints/best_1-1.pt
```

### 6.4. Đánh giá mô hình theo chuẩn Mục 4
```bash
python evaluate.py --checkpoint checkpoints/best_1-1.pt --env-id SuperMarioBros-1-1-v0 --num-episodes 10
```

### 6.5. Vẽ biểu đồ học tập & Phân tích thất bại
```bash
# Vẽ đường cong học tập
python analysis/plot_curves.py --log-file logs/progress.csv --output analysis/learning_curves.png

# Phân tích phân bố tọa độ x kết thúc lượt chơi
python analysis/plot_failure_analysis.py --eval-json eval_results/eval_best_1-1_SuperMarioBros-1-1-v0.json
```

### 6.6. Đóng gói nộp bài
```bash
python package_submission.py --group 01
```

---

## 7. Chiến lược quản trị rủi ro & Git Workflow

### 7.1. Nguyên tắc làm việc trên Git
- Mỗi thành viên tạo nhánh riêng: `feature/sv1-ppo`, `feature/sv2-env-gae`, `feature/sv3-model-eval`.
- Commit rõ ràng từng hàm để lịch sử commit khớp chính xác với bảng phân công công việc trong báo cáo.
- Thực hiện Code Review chéo trước khi merge vào nhánh `main` trước tối 20/11.

### 7.2. Xử lý các rủi ro đã nhận diện
1. **Tiến độ cài đặt gấp (4 ngày):** Nếu trễ hạn cài đặt, bắt buộc hoàn thành và chốt kiểm thử trong sáng 21/11 để kịp bật máy chạy huấn luyện.
2. **Tốc độ huấn luyện:** Đo số bước trên giây (`sps`) trong lần chạy đầu. Nếu tốc độ ~100 sps (cần ~28 giờ cho 10M bước), tính toán lại số bước mục tiêu phù hợp với cấu hình máy.
3. **Chạy hỏng:** Theo dõi kỹ chẩn đoán ở mốc 500k bước để dừng sớm và điều chỉnh siêu tham số, tránh lãng phí 1-2 ngày chạy máy vô ích.
4. **Màn thưởng:** Đảm bảo sử dụng học chuyển giao (Transfer Learning) từ checkpoint 1-1 đã hội tụ thay vì train từ đầu để rút ngắn thời gian.
