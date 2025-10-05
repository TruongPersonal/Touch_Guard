# Touch Guard
TouchGuard là demo nhận diện chạm tay lên mặt chạy trực tiếp trên trình duyệt bằng TensorFlow.js. Ứng dụng dùng MobileNet để trích xuất đặc trưng và KNN Classifier để phân loại 2 trạng thái: Không chạm / Có chạm. Mọi xử lý đều chạy cục bộ trên máy bạn.

## Tính năng chính
Huấn luyện nhanh 2 lớp bằng webcam (không cần dữ liệu sẵn).
Cảnh báo bằng âm thanh + thông báo trình duyệt khi phát hiện chạm.
Lưu mô hình gọn (centroid) vào IndexedDB → mở lại vẫn dùng được.
Chọn camera, điều chỉnh ngưỡng, tắt/bật âm, hiển thị FPS.

## Yêu cầu
Trình duyệt hỗ trợ WebGL/WebGPU (Chrome/Edge/Brave mới).
HTTPS hoặc http://localhost để cấp quyền camera.

## Cách dùng nhanh
Mở app, cho phép truy cập camera.
Bước 1: nhấn Train 1 (KHÔNG chạm tay).
Bước 2: nhấn Train 2 (CÓ chạm tay).
Nhấn Run để bắt đầu cảnh báo.
Có thể chỉnh Ngưỡng (Confidence), Tắt âm, Batches trong phần giao diện.

## Lưu trữ & quyền riêng tư
Dữ liệu mô hình lưu ở IndexedDB với key: touch-guard-dataset-v1.
Trạng thái tắt âm lưu ở localStorage key: tg-muted.
Ứng dụng không gửi hình ảnh/âm thanh ra ngoài; mọi suy luận chạy trên thiết bị.

## Sửa lỗi nhanh
Không thấy camera: kiểm tra HTTPS/localhost và quyền Camera trong trình duyệt.
Âm vẫn phát: tắt bằng công tắc “Tắt âm”; trạng thái được lưu và giữ khi mở lại.
Muốn huấn luyện lại: nhấn Reset để xóa mẫu + IndexedDB.

## Công nghệ
React, TensorFlow.js, MobileNet, KNN Classifier
idb-keyval (IndexedDB), howler.js (âm thanh)

## Giấy phép
MIT — dùng tự do cho học tập & demo.