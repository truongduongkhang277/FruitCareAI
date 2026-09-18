"""FruitCare AI: local Gradio app, adapted from the supplied notebook."""
import html
import mimetypes
mimetypes.add_type("image/webp", ".webp")
import os
from common import load_settings, load_classifier, image_batch
load_settings()
model, class_names, model_error = None, [], ""
try:
    model, class_names = load_classifier()
except Exception as exc:
    model_error = f"{type(exc).__name__}: {exc}"
    print(model_error)

from claude_chat import ask_claude

# =====================================================================
# 12. FRUITCARE AI - GIAO DIỆN DỰ ÁN HOÀN CHỈNH
# =====================================================================
# Giao diện: logo + upload ảnh + kết quả Fresh/Rotten tiếng Việt + ChatBot AI
# ChatBot dùng context của kết quả phân loại gần nhất.

import os
import json
import numpy as np
import gradio as gr
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------
# 1) ÁNH XẠ NHÃN MÔ HÌNH -> TIẾNG VIỆT
# ---------------------------------------------------------------------
FRUIT_MAP_VI = {
    "apple": "Táo", "apples": "Táo",
    "banana": "Chuối", "bananas": "Chuối",
    "orange": "Cam", "oranges": "Cam",
}

def parse_fruit_label(raw_label):
    label = str(raw_label).lower().strip()
    if label.startswith("fresh"):
        condition_key = "fresh"
        condition_vi = "Tươi"
        fruit_key = label[5:]
    elif label.startswith("rotten"):
        condition_key = "rotten"
        condition_vi = "Hư / Thối"
        fruit_key = label[6:]
    else:
        condition_key = "unknown"
        condition_vi = "Không xác định"
        fruit_key = label
    fruit_vi = FRUIT_MAP_VI.get(fruit_key, fruit_key.capitalize())
    return fruit_vi, condition_vi, condition_key

# ---------------------------------------------------------------------
# 2) TRẠNG THÁI PHÂN LOẠI DÙNG CHUNG CHO CHATBOT
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 3) HÀM ĐỌC ẢNH UPLOAD
# ---------------------------------------------------------------------
def load_uploaded_image(filepath):
    if not filepath:
        return None
    try:
        with Image.open(filepath) as source:
            source.seek(0)  # Use the first frame of animated WebP/GIF.
            img = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, img).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise gr.Error("Không đọc được ảnh. Kiểm tra file có mở được trên máy và Pillow hỗ trợ định dạng này.") from exc

# ---------------------------------------------------------------------
# 4) HÀM PHÂN LOẠI
# ---------------------------------------------------------------------
def classify_fruit_for_pro_ui(img):
    if img is None:
        return None, "", "", "", "", "Vui lòng tải ảnh trái cây lên trước.", ""

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img).astype("uint8"))

    original = img.convert("RGB")
    batch = image_batch(original)

    # Đưa pixel 0–255 vào model; EfficientNet có tiền xử lý tích hợp.
    if model is None:
        raise gr.Error(model_error)
    predictions = model(batch, training=False).numpy()[0]
    idx = int(np.argmax(predictions))
    raw_label = class_names[idx]
    confidence = float(predictions[idx]) * 100.0
    fruit_vi, condition_vi, condition_key = parse_fruit_label(raw_label)

    top_indices = np.argsort(predictions)[::-1][:3]
    top_rows = []
    for rank, i in enumerate(top_indices, start=1):
        f_vi, c_vi, _ = parse_fruit_label(class_names[int(i)])
        top_rows.append(f"{rank}. {f_vi} — {c_vi}: {float(predictions[int(i)])*100:.2f}%")
    top3_text = "\n".join(top_rows)

    badge = "🟢 TƯƠI" if condition_key == "fresh" else ("🔴 HƯ / THỐI" if condition_key == "rotten" else "🟡 KHÔNG XÁC ĐỊNH")
    advice = (
        "Có thể sử dụng nếu kiểm tra thực tế không có dấu hiệu bất thường."
        if condition_key == "fresh" else
        "Nên ưu tiên an toàn thực phẩm; kiểm tra kỹ và không sử dụng nếu có mốc, mùi lạ, nhớt hoặc phân hủy rõ rệt."
        if condition_key == "rotten" else
        "Hãy thử một ảnh rõ hơn, đủ sáng và tập trung vào trái cây."
    )

    result_html = f"""
    <div class='result-card {condition_key}'>
      <div class='result-badge'>{badge}</div>
      <div class='result-fruit'>{html.escape(fruit_vi)}</div>
      <div class='result-confidence'>Độ tin cậy <b>{confidence:.2f}%</b></div>
      <div class='result-advice'>{advice}</div>
    </div>
    """

    context = (
        f"Kết quả mô hình: trái cây = {fruit_vi}; tình trạng = {condition_vi}; "
        f"nhãn kỹ thuật = {raw_label}; độ tin cậy = {confidence:.2f}%."
    )

    return original, result_html, fruit_vi, condition_vi, f"{confidence:.2f}%", top3_text, context

# ---------------------------------------------------------------------
# 5) CHATBOT CLAUDE CODE
# ---------------------------------------------------------------------
SYSTEM_INSTRUCTION_PRO = """
Bạn là FruitCare AI, trợ lý tiếng Việt của hệ thống nhận diện trái cây.
Hệ thống dùng EfficientNetB0 để phân loại 6 lớp: táo/chuối/cam ở trạng thái tươi hoặc hư.

Quy tắc:
- Luôn ưu tiên tiếng Việt, trả lời thân thiện và dễ hiểu.
- Dùng kết quả mô hình được cung cấp làm ngữ cảnh, nhưng không coi dự đoán AI là tuyệt đối.
- Nếu độ tin cậy thấp, nói rõ cần ảnh tốt hơn hoặc kiểm tra thực tế.
- Với trái cây bị hư/thối, ưu tiên an toàn thực phẩm; không khuyến khích ăn phần còn lại nếu có mốc, mùi bất thường, nhớt hoặc phân hủy rõ rệt.
- Có thể tư vấn bảo quản, giải thích độ tin cậy, ý nghĩa Fresh/Rotten và cách cải thiện ảnh.
- Chỉ nhận kết quả phân loại dạng văn bản, không trực tiếp xem ảnh trong cuộc trò chuyện này.
- Mô hình chỉ biết táo/chuối/cam; ảnh trái cây khác vẫn có thể bị gán nhãn với điểm cao.
- Nếu người dùng cho biết quả khác, không khẳng định nhãn mô hình là đúng.
- Không bịa ra thông tin mà mô hình không cung cấp.
"""

def fruit_chat_pro(message, history, prediction_context):
    if not message or not str(message).strip():
        return history or [], ""

    history = history or []
    context = prediction_context.strip() if prediction_context else "Chưa có ảnh nào được phân loại trong phiên này."

    recent = []
    for item in history[-8:]:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            if isinstance(content, str):
                recent.append(f"{role}: {content}")

    prompt = f"""
NGỮ CẢNH KẾT QUẢ NHẬN DIỆN:
{context}

LỊCH SỬ HỘI THOẠI:
{chr(10).join(recent) if recent else '(chưa có)'}

CÂU HỎI:
{message}
"""

    try:
        answer = ask_claude(SYSTEM_INSTRUCTION_PRO + "\n\n" + prompt)
    except RuntimeError as exc:
        answer = str(exc)

    new_history = history + [
        {"role": "user", "content": str(message)},
        {"role": "assistant", "content": answer},
    ]
    return new_history, ""

# ---------------------------------------------------------------------
# 6) CSS - PHONG CÁCH DASHBOARD CHUYÊN NGHIỆP
# ---------------------------------------------------------------------
CUSTOM_CSS = """
:root { --fc-green:#16a34a; --fc-dark:#10231a; --fc-soft:#f3faf5; --fc-border:#dcebe1; }
.gradio-container { max-width: 1420px !important; margin: auto !important; background: #f7faf8 !important; }
#hero { background: linear-gradient(135deg,#0f5132,#16a34a); border-radius: 24px; padding: 28px 34px; color: white; box-shadow: 0 12px 35px rgba(16,81,50,.18); margin-bottom: 18px; }
#hero h1 { margin: 0; font-size: 34px; letter-spacing: -.5px; }
#hero p { margin: 7px 0 0; opacity: .92; font-size: 15px; }
.logo-mark { width: 54px; height:54px; border-radius: 16px; background: rgba(255,255,255,.18); display:flex; align-items:center; justify-content:center; font-size:30px; margin-bottom:12px; }
.panel { background:white; border:1px solid var(--fc-border); border-radius:20px; padding:18px; box-shadow: 0 5px 20px rgba(15,81,50,.06); }
.section-title { font-size:18px; font-weight:700; color:var(--fc-dark); margin-bottom:10px; }
#upload-zone { border:2px dashed #b9d8c3 !important; border-radius:18px !important; background:#fbfefc !important; }
.result-card { border-radius:18px; padding:22px; margin-top:10px; border:1px solid #dcebe1; background:#fbfefc; }
.result-card.fresh { border-left:6px solid #16a34a; }
.result-card.rotten { border-left:6px solid #dc2626; }
.result-badge { font-size:14px; font-weight:800; letter-spacing:.5px; }
.result-fruit { font-size:30px; font-weight:800; color:#10231a; margin:6px 0; }
.result-confidence { color:#557064; font-size:14px; }
.result-advice { margin-top:14px; padding:12px 14px; border-radius:12px; background:#f3faf5; color:#294538; font-size:14px; line-height:1.5; }
#chat-panel { min-height: 640px; }
footer { text-align:center; color:#6b7f74; font-size:12px; padding:12px; }
"""

# ---------------------------------------------------------------------
# 7) GIAO DIỆN
# ---------------------------------------------------------------------
with gr.Blocks(title="FruitCare AI", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    prediction_context_state = gr.State("")
    if model_error:
        gr.Markdown("**Chưa sẵn sàng nhận diện:** " + model_error)
    gr.HTML("""
    <div id='hero'>
      <div class='logo-mark'>🍏</div>
      <h1>FruitCare AI</h1>
      <p>Hệ thống nhận diện trái cây Tươi / Hư bằng EfficientNetB0 kết hợp trợ lý AI.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # ================= LEFT: IMAGE =================
        with gr.Column(scale=5, elem_classes="panel"):
            gr.Markdown("### 📷 Phân tích trái cây", elem_classes="section-title")
            upload_btn = gr.UploadButton(
                "📤  Chọn ảnh từ máy tính",
                file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"], file_count="single",
                variant="primary", size="lg"
            )
            input_image = gr.Image(
                type="pil", label="Ảnh trái cây", sources=["upload", "webcam"],
                elem_id="upload-zone", height=330
            )
            # Nút upload đưa ảnh vào vùng preview; sau đó tự động phân tích.
            upload_event = upload_btn.upload(load_uploaded_image, upload_btn, input_image)

            with gr.Row():
                analyze_btn = gr.Button("🔎 PHÂN TÍCH ẢNH", variant="primary", size="lg")
                clear_btn = gr.Button("↺ Xóa", variant="secondary", size="lg")

            result_html = gr.HTML("<div class='result-card'><div class='result-badge'>🟡 CHƯA PHÂN TÍCH</div><div class='result-fruit'>Tải ảnh để bắt đầu</div></div>")

            with gr.Row():
                fruit_output = gr.Textbox(label="Loại trái cây", interactive=False)
                condition_output = gr.Textbox(label="Tình trạng", interactive=False)
                confidence_output = gr.Textbox(label="Độ tin cậy", interactive=False)

            top3_output = gr.Textbox(label="Top 3 dự đoán", lines=4, interactive=False)

        # ================= RIGHT: CHATBOT =================
        with gr.Column(scale=6, elem_classes="panel"):
            gr.Markdown("### 🤖 FruitCare AI Assistant", elem_classes="section-title")
            gr.Markdown("ChatBot tự động nhận kết quả phân loại gần nhất làm ngữ cảnh. Bạn có thể hỏi về kết quả, bảo quản hoặc ý nghĩa độ tin cậy.")
            chatbot = gr.Chatbot(
                value=[{"role":"assistant", "content":"Xin chào! 👋 Hãy tải ảnh trái cây ở bên trái và bấm **Phân tích ảnh**. Sau đó bạn có thể hỏi mình về kết quả."}],
                type="messages", height=470, label="Trợ lý AI"
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ví dụ: Quả này có nên ăn không?",
                    show_label=False, scale=8, container=True
                )
                send_btn = gr.Button("Gửi ➤", variant="primary", scale=2)
            gr.Examples(
                examples=[
                    ["Giải thích kết quả phân loại vừa rồi."],
                    ["Độ tin cậy của mô hình có ý nghĩa gì?"],
                    ["Trái cây này nên bảo quản như thế nào?"],
                    ["Nếu quả bị hư thì có nên cắt phần hư rồi ăn không?"],
                ], inputs=chat_input
            )

    analyze_btn.click(
        classify_fruit_for_pro_ui,
        inputs=input_image,
        outputs=[input_image, result_html, fruit_output, condition_output, confidence_output, top3_output, prediction_context_state]
    )

    upload_event.success(
        classify_fruit_for_pro_ui,
        inputs=input_image,
        outputs=[input_image, result_html, fruit_output, condition_output, confidence_output, top3_output, prediction_context_state]
    )

    clear_btn.click(
        lambda: (None, "<div class='result-card'><div class='result-badge'>🟡 CHƯA PHÂN TÍCH</div><div class='result-fruit'>Tải ảnh để bắt đầu</div></div>", "", "", "", "", ""),
        outputs=[input_image, result_html, fruit_output, condition_output, confidence_output, top3_output, prediction_context_state]
    )

    input_image.input(
        lambda: ("", "", "", "", "", "", []),
        outputs=[result_html, fruit_output, condition_output, confidence_output,
                 top3_output, prediction_context_state, chatbot]
    )
    clear_btn.click(lambda: [], outputs=chatbot)
    upload_btn.upload(lambda: [], outputs=chatbot)

    send_btn.click(fruit_chat_pro, [chat_input, chatbot, prediction_context_state], [chatbot, chat_input])
    chat_input.submit(fruit_chat_pro, [chat_input, chatbot, prediction_context_state], [chatbot, chat_input])

    gr.HTML("<footer>FruitCare AI • EfficientNetB0 + Claude Code • Hệ thống hỗ trợ tham khảo, cần kiểm tra thực tế trước khi sử dụng thực phẩm.</footer>")

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
