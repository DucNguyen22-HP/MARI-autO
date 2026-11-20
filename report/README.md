# Báo cáo Đồ án (Tối đa 8 trang PDF)

### Phân công viết báo cáo (Bắt đầu từ 22/11, hoàn thiện 1/12 - 2/12):
- **SV1**: Thiết lập huấn luyện, giải thích PPO, thí nghiệm so sánh.
- **SV2**: Chi tiết cài đặt thuật toán & môi trường, bảng tổng hợp kết quả pytest, định hình phần thưởng (reward shaping).
- **SV3**: Phân tích đường cong học tập, phân tích thất bại (vị trí x cuối lượt), tổng hợp và ghép file PDF báo cáo hoàn chỉnh.

### Cấu trúc bài báo cáo:
1. **Giới thiệu**: Đặt vấn đề bài toán tự động chơi Super Mario Bros với Học tăng cường (PPO).
2. **Kiến trúc mạng & Môi trường**: Mô tả ActorCritic CNN, tiền xử lý (FrameSkip, GrayResize), định hình phần thưởng (ScalarizeReward).
3. **Thuật toán**: Giải thích chi tiết PPO Clip loss, GAE ($\lambda$), tối ưu hóa và siêu tham số.
4. **Kiểm thử đơn vị & Tích hợp**: Bảng kết quả chạy pytest toàn diện (`tests/`).
5. **Thí nghiệm & Đánh giá**:
   - Kết quả huấn luyện màn 1-1 (learning curves, flag_rate, x_pos trung bình).
   - Thí nghiệm so sánh (Ablation study $\gamma = 0.9$ vs $0.99$, seed thứ 2).
   - Kết quả học chuyển giao (Transfer learning) trên màn thưởng (1-2 / 3-1).
6. **Phân tích thất bại & Thảo luận**: Biểu đồ phân bố điểm chết/vướng của Mario.
7. **Kết luận**: Đánh giá kết quả so với mục tiêu và bài học kinh nghiệm.
