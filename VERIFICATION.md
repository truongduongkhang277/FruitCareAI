# Phạm vi kiểm tra

Môi trường kiểm tra: Linux, Python 3.12, TensorFlow CPU 2.18.0, Keras 3.15.1,
Gradio 5.50.0. Hướng dẫn cài đặt cho người dùng nhắm tới Windows/Python 3.11.

Đã thực hiện:

- Biên dịch kiểm tra cú pháp các tệp Python.
- Nạp lại model `.keras`, kiểm tra nhãn và kích thước input.
- Nạp H5 mẫu có `Lambda(preprocess_input)` giống kiểu lưu của notebook.
- Xử lý ảnh RGB, resize 224×224, giữ pixel 0–255.
- Tạo Gradio Blocks, gọi hàm phân loại, xử lý chưa chọn ảnh, chatbot không có API key.
- Khởi động ứng dụng và nhận HTTP 200 từ trang localhost.
- Chạy `predict.py`, `evaluate.py` trên model nhỏ và ảnh tổng hợp; tạo báo cáo và confusion matrix.
- Chạy xuất TFLite cho model mẫu, so sánh đầu ra với Keras.
- Chạy forward pass EfficientNetB0 thật với trọng số ngẫu nhiên.
- Chạy nguyên luồng `train.py` với 24 ảnh tổng hợp, 1 epoch mỗi giai đoạn;
  chỉ thay ImageNet bằng `weights=None` trong phép thử. Đã lưu và nạp lại checkpoint,
  kiểm tra JSON lịch sử hai epoch và biểu đồ. Đây là kiểm tra hoạt động, không đo chất lượng.
- Kiểm tra giải nén ZIP hợp lệ và từ chối đường dẫn thoát khỏi thư mục đích.

Giới hạn:

- Chưa chạy trên Windows/VS Code thực tế; các file BAT và cấu hình VS Code đã được rà soát.
- Chưa có dataset/model trái cây đã huấn luyện của người gửi; không báo cáo độ chính xác thực tế.
- Chưa gọi Kaggle/Gemini bằng tài khoản thật, chưa tải trọng số ImageNet trong phép thử.
- Xuất TFLite đã thử bằng model mẫu, chưa xác nhận model trái cây huấn luyện đầy đủ.
- Không đưa model/ảnh giả dùng kiểm tra vào bản phát hành để tránh nhầm thành model sử dụng thật.
