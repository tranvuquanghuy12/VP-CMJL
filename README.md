# VP-CMJL: Visual Proxy - Compositional Multi-Job Learning

Dự án này triển khai mô hình **VP-CMJL** cho bài toán **Compositional Zero-Shot Learning (CZSL)** trên bộ dữ liệu hoa quả Việt Nam (**tlu-states**). [cite: 2026-01-13]

## 📊 Kết quả đánh giá (Metrics)
Trong bài toán CZSL, chúng tôi sử dụng 4 chỉ số chính để đánh giá mô hình: [cite: 2026-01-13]

* **Seen (S)**: Khả năng nhận diện các cặp (Thuộc tính - Đối tượng) đã xuất hiện trong quá trình huấn luyện. [cite: 2026-01-13]
* **Unseen (U)**: Khả năng suy luận trên các cặp mới hoàn toàn mà mô hình chưa từng thấy. [cite: 2026-01-13]
* **Harmonic Mean (HM)**: Chỉ số trung bình điều hòa giữa S và U, đánh giá thực lực tổng thể của mô hình. [cite: 2026-01-13]
    * Công thức: $$HM = \frac{2 \cdot S \cdot U}{S + U}$$
* **AUC (Area Under Curve)**: Diện tích dưới đường cong độ chính xác, thể hiện độ ổn định của mô hình khi thay đổi các ngưỡng bias. [cite: 2026-01-13]

## 🚀 Cài đặt và Chạy
1. Clone dự án và cài đặt môi trường.
2. Huấn luyện: `python train_multi_proxy.py --dataset tlu-states`
3. Chấm điểm: `python test_multi_proxy.py --dataset tlu-states --load_model [path_to_weights]` [cite: 2026-01-13]