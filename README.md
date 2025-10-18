# TouchGuard
TouchGuard là ứng dụng nhận diện **chạm tay lên mặt** chạy trực tiếp trên trình duyệt bằng **TensorFlow.js**.
Ứng dụng sử dụng **MobileNet** để trích xuất đặc trưng ảnh và **KNN Classifier** để phân loại hai trạng thái: **Không chạm** và **Có chạm**.
Toàn bộ xử lý được thực hiện **cục bộ trên thiết bị** của bạn, không gửi dữ liệu ra ngoài.

---

## ⚙️ Cách hoạt động
- **MobileNet** trích xuất vector đặc trưng (embedding) từ khung hình webcam.
- **KNN Classifier** học từ mẫu bạn huấn luyện (chạm / không chạm).
- Khi chạy, ứng dụng so sánh vector hiện tại với mẫu đã học để dự đoán hành động.
- Nếu phát hiện chạm, hệ thống **phát âm thanh cảnh báo** và **gửi thông báo trình duyệt**.

---

## 🚀 Cách sử dụng
1. Mở ứng dụng, **cho phép quyền truy cập camera**.
2. Nhấn **Train 1** (khi không chạm tay).
3. Nhấn **Train 2** (khi có chạm tay).
4. Nhấn **Run** để bắt đầu nhận diện.

### Tuỳ chọn
- **Ngưỡng (Confidence):** điều chỉnh độ chắc chắn của dự đoán.
- **Tắt âm:** bật/tắt âm thanh cảnh báo.
- **Batches:** huấn luyện nhiều vòng hơn để tăng độ chính xác.

---

## 💾 Lưu trữ
- Dữ liệu mô hình: **IndexedDB** (`touch-guard-dataset-v1`).
- Trạng thái tắt âm: **localStorage** (`tg-muted`).
- Có thể mở lại ứng dụng mà không cần huấn luyện lại.

---

## 🔧 Sửa lỗi thường gặp
- **Không thấy camera:** kiểm tra quyền truy cập camera hoặc bật HTTPS/localhost.
- **Âm thanh không tắt:** bật công tắc “Tắt âm”.
- **Muốn huấn luyện lại:** nhấn **Reset** để xoá toàn bộ dữ liệu huấn luyện.

---

## 🧠 Công nghệ sử dụng
- **React**
- **TensorFlow.js**
- **MobileNet + KNN Classifier**
- **IndexedDB (idb-keyval)**
- **howler.js (âm thanh)**

---

## 📜 Giấy phép
**MIT License** — dùng tự do cho học tập, nghiên cứu, và demo.