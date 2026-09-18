# FruitCare AI — chạy bằng Visual Studio Code

Bản chuyển đổi từ `2500022593_GiangVinhKiet_FruitCareAI_Pro.ipynb`.
Giữ EfficientNetB0, huấn luyện hai giai đoạn, nhận diện ảnh, giao diện tiếng Việt
và chatbot Gemini; tách thành chương trình Python chạy trên máy cá nhân.

## 1. Bắt đầu trên Windows

1. Cài **Python 3.11 bản 64-bit**. Bộ phụ thuộc này dùng TensorFlow 2.18;
   không dùng Python 3.14 cho môi trường của dự án.
2. Giải nén toàn bộ ZIP, ví dụ vào `D:\Project\FruitCareAI_VSCode`.
3. VS Code → **File → Open Folder** → chọn thư mục có `app.py` và `requirements.txt`.
4. Cài extension **Python** của Microsoft; chấp nhận extension Python Debugger nếu được đề nghị.
5. Chạy `setup_windows.bat` bằng cách nhấp đúp trong File Explorer.
   Cần Internet để tải thư viện; dung lượng cài đặt TensorFlow khá lớn.
6. Trong VS Code, nhấn **Ctrl+Shift+P → Python: Select Interpreter**,
   chọn `.venv\Scripts\python.exe` của thư mục vừa giải nén.
7. Mở **Terminal → New Terminal**. Những lệnh dưới dùng Python của `.venv`
   trực tiếp, không cần đổi ExecutionPolicy hoặc kích hoạt PowerShell script.

Có thể thay bước 5 bằng các lệnh PowerShell, chạy từng dòng:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Không chép đè `.env` nếu đã điền cấu hình trước đó.

## 2. Nếu đã có model huấn luyện từ Colab

Notebook là mã nguồn và kết quả hiển thị, **không phải tệp trọng số**.
Bản tải xuống này không kèm dataset, model đã train hoặc API key.

Chép hai tệp cùng một lần huấn luyện vào thư mục `models`:

- `fruit_classifier_model.h5` hoặc `fruit_classifier_model.keras`
- `fruit_class_names.json`

Nếu model là `.h5`, sửa `.env`:

```dotenv
MODEL_PATH=models/fruit_classifier_model.h5
LABELS_PATH=models/fruit_class_names.json
GEMINI_API_KEY=
GEMINI_MODEL=
```

Chạy ứng dụng:

```powershell
.\.venv\Scripts\python.exe app.py
```

Hoặc nhấp đúp `run_app.bat`, hoặc nhấn F5 trong VS Code và chọn
**FruitCareAI: mở ứng dụng**. Mở http://127.0.0.1:7860 nếu trình duyệt chưa tự bật.
Dùng nút chọn ảnh hoặc camera, rồi phân tích ảnh. Ctrl+C trong terminal để dừng.
Ứng dụng chỉ lắng nghe máy cục bộ, không tạo đường dẫn Gradio công khai.

Model H5 cũ được nạp với `custom_objects` cho `preprocess_input`. Nếu gặp lỗi
khác do phiên bản Keras/định dạng Lambda, hãy xuất lại model từ môi trường Colab
ban đầu hoặc huấn luyện lại bằng `train.py`. Không tự đổi đuôi H5 thành Keras.
JSON nhãn bắt buộc phải khớp thứ tự lớp của model; không dùng nhãn dự phòng đoán sẵn.

Nếu chưa có model, giao diện vẫn mở và hiển thị hướng dẫn; chức năng phân loại
chỉ hoạt động sau khi hoàn thành bước 3–4 hoặc bổ sung hai tệp ở trên.

## 3. Chuẩn bị dữ liệu khi chưa có model

Bộ dữ liệu gốc:
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

**Cách A — tải ZIP bằng trình duyệt:** lưu file ZIP vào thư mục dự án rồi chạy:

```powershell
.\.venv\Scripts\python.exe download_data.py --zip fruits-fresh-and-rotten-for-classification.zip
```

**Cách B — Kaggle API:** lấy `kaggle.json` từ tài khoản Kaggle và đặt tại
`C:\Users\<ten-user>\.kaggle\kaggle.json`, sau đó chạy:

```powershell
.\.venv\Scripts\python.exe download_data.py
```

Script in ra thư mục tìm thấy có `train` và `test`. Cấu trúc mặc định mong đợi:

```text
fruit_dataset/dataset/train/freshapples/
fruit_dataset/dataset/train/freshbananas/
fruit_dataset/dataset/train/freshoranges/
fruit_dataset/dataset/train/rottenapples/
fruit_dataset/dataset/train/rottenbananas/
fruit_dataset/dataset/train/rottenoranges/
fruit_dataset/dataset/test/<các thư mục lớp tương ứng>/
```

Nếu giải nén ra đường dẫn lồng khác, truyền đường dẫn đúng bằng `--data`.
Không cần di chuyển ảnh để ép đúng tên thư mục mặc định.

## 4. Huấn luyện

```powershell
.\.venv\Scripts\python.exe train.py
```

Mặc định: batch 8, 8 epoch đóng băng backbone, 15 epoch fine-tuning 50 lớp cuối
(trừ BatchNormalization), seed 42. Lần đầu cần Internet tải trọng số ImageNet.
Validation lấy 20% từ `train`; `test` chỉ dùng khi đánh giá cuối cùng.
Không cache toàn bộ ảnh vào RAM. Dùng batch 4 nếu thiếu bộ nhớ:

```powershell
.\.venv\Scripts\python.exe train.py --data "fruit_dataset/dataset" --batch-size 4
```

Thử luồng với số epoch nhỏ, **không dùng kết quả này làm mô hình hoàn chỉnh**:

```powershell
.\.venv\Scripts\python.exe train.py --epochs-head 1 --epochs-fine 0 --output models_trial
```

Model tốt nhất theo validation loss được lưu vào:

- `models/fruit_classifier_model.keras`
- `models/fruit_class_names.json`
- `models/history.json`, `models/training_config.json`
- `models/training_history.png`

Mỗi thử nghiệm mới dùng `--output` riêng. Khi dùng thư mục khác `models`, sửa
MODEL_PATH/LABELS_PATH trong `.env`, hoặc truyền `--model` và `--labels` cho
các lệnh predict/evaluate/export. Huấn luyện bằng CPU có thể mất nhiều thời gian;
không có ước tính thời lượng chính xác khi chưa đo trên máy thực tế.

## 5. Kiểm tra và xuất mô hình

Đánh giá tập test, xuất JSON chỉ số, precision/recall/F1, confusion matrix và
thời gian suy luận thực đo (20 lần, batch 1, có warmup):

```powershell
.\.venv\Scripts\python.exe evaluate.py --data fruit_dataset/dataset/test
```

Kết quả nằm ở `outputs/evaluation.json` và `outputs/confusion_matrix.png`.
Thời gian đo không bao gồm đọc/resize ảnh hoặc gọi Gemini.
Nếu dùng model Colab cũ từng chọn mô hình bằng tập test này, kết quả không được
coi là phép đo độc lập hoàn toàn; cần tập test mới để đánh giá khách quan.

Dự đoán ảnh không mở giao diện:

```powershell
.\.venv\Scripts\python.exe predict.py "D:\Anh\tao.jpg" "D:\Anh\chuoi.png"
```

Xuất định dạng bổ sung:

```powershell
.\.venv\Scripts\python.exe export_model.py --format h5
.\.venv\Scripts\python.exe export_model.py --format tflite
```

TFLite dùng input float32 `[1,224,224,3]`, pixel 0–255; đi kèm JSON nhãn.
Script kiểm tra đầu ra TFLite với Keras bằng một tensor mẫu trước khi ghi file.
Bước chuyển đổi tách riêng, nếu thất bại không ảnh hưởng model Keras đã lưu.
Exporter dùng API nội bộ TensorFlow để đóng băng graph; giữ TensorFlow 2.18
như requirements khi sử dụng exporter này.

## 6. Chatbot Gemini (tùy chọn)

Mở `.env` và điền:

```dotenv
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=MODEL_ID_KHA_DUNG_TRONG_TAI_KHOAN
```

Lấy đúng ID từ Google AI Studio/tài khoản đang dùng. Không giữ cứng tên
`gemini-3.7-flash` của notebook vì quyền truy cập/model có thể thay đổi.
Khởi động lại `app.py` sau khi sửa `.env`. Nếu để trống, phần phân loại ảnh vẫn chạy.
Chatbot gửi nội dung hỏi, lịch sử gần nhất và kết quả phân loại dạng văn bản tới
Gemini; ảnh không được gửi tới Gemini bằng mã này. Không đưa `.env`/API key vào Git.

## 7. Nội dung đã rà soát và sửa

| Vấn đề trong notebook | Bản VS Code |
|---|---|
| `google.colab`, upload/secrets và lệnh `!pip/!unzip` | Tệp local, `.env`, requirements và script tải/giải nén |
| Dùng `test` làm validation rồi gọi là test độc lập | Validation tách từ train; evaluate đọc test riêng |
| `cache()` toàn ảnh và shuffle tới 1.000 batch | Không cache RAM; prefetch 1; batch mặc định 8 |
| Chú thích EfficientNet chuẩn hóa về −1..1 | Sửa thành pixel 0–255; bỏ Lambda thừa ở model mới |
| Giả lập accuracy/loss CNN, ResNet và nhân tỷ lệ thời gian | Bỏ biểu đồ giả lập; chỉ lưu số đo thực của model được nạp |
| Nhãn dự phòng khi thiếu JSON | Báo thiếu file, kiểm tra số lớp và input model |
| Lỗi thiếu model vẫn chạy phần suy luận | Hiển thị trạng thái thiếu model, chặn suy luận |
| Gemini bắt buộc và model ID cố định | Gemini tùy chọn, cấu hình qua `.env` |
| State nằm ngoài Blocks, ngữ cảnh có thể cũ sau đổi ảnh | State trong Blocks; xóa kết quả và lịch sử khi đổi ảnh |
| `share=True` | Chạy localhost, `share=False` |
| Lưu model cuối dù epoch trước tốt hơn | Lưu checkpoint có validation loss thấp nhất qua hai giai đoạn |

Bản này không huấn luyện CNN cơ bản/ResNet50 vì notebook gốc cũng không có phần
huấn luyện hai mô hình đó. Không có số liệu so sánh thực nghiệm được bịa thêm.
Việc chia validation ngẫu nhiên chưa kiểm soát ảnh trùng/cùng một quả giữa các tập;
nếu dùng cho nghiên cứu, cần kiểm tra dữ liệu và chia theo nhóm ảnh/đối tượng.

## 8. Xử lý lỗi thường gặp

- `No module named ...`: chọn đúng interpreter `.venv`, cài lại requirements bằng
  chính `.venv\Scripts\python.exe`.
- `No matching distribution found`: kiểm tra `py -3.11 --version`, dùng Python 3.11
  64-bit và tạo môi trường mới; không dùng lại venv Python 3.14.
- `DLL load failed` khi import TensorFlow: kiểm tra Python 64-bit và Microsoft
  Visual C++ Redistributable x64 theo hướng dẫn TensorFlow.
- Không có GPU khi chạy Windows trực tiếp: bản TensorFlow này dùng CPU trên Windows
  native. Muốn CUDA cần môi trường Linux/WSL2 tương thích; cài PyTorch CUDA không làm
  TensorFlow tự dùng GPU. Không cần GPU để mở ứng dụng và suy luận.
- Kaggle trả 401/403: kiểm tra thông tin Kaggle hoặc dùng cách tải ZIP bằng trình duyệt.
- Thiếu thư mục train/test: dùng đường dẫn mà `download_data.py` in ra.
- Cổng 7860 đang dùng: dừng phiên ứng dụng cũ hoặc sửa `server_port` cuối `app.py`.
- Gemini báo lỗi: kiểm tra ID model, API key, quyền truy cập và hạn mức tài khoản.

## Tài liệu kỹ thuật tham khảo

- TensorFlow: https://www.tensorflow.org/install/pip
- Keras EfficientNet (input 0–255, preprocessing tích hợp): https://keras.io/api/applications/efficientnet/
- Gemini model catalog: https://ai.google.dev/gemini-api/docs/models

Xem `VERIFICATION.md` để biết phạm vi đã kiểm tra trước khi đóng gói.

Mô hình chỉ học táo, chuối, cam tươi/hư. Ảnh ngoài sáu lớp vẫn có thể bị gán một
nhãn với điểm cao; điểm softmax không phải xác suất bảo đảm thực phẩm an toàn.
