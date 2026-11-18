import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
INVOICES_DIR = BASE_DIR / "invoices"
HISTORY_FILE = BASE_DIR / "history.json"
INVOICES_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.3
INPUT_SIZE = (224, 224)

FOOD_LABELS = [
    "Ca hu kho",
    "Canh cai",
    "Canh chua",
    "Com trang",
    "Dau hu sot ca",
    "Ga chien",
    "Rau muong xao",
    "Thit kho",
    "Thit kho trung",
    "Trung chien",
]

PRICE_LIST = {
    "Ca hu kho": 22000,
    "Canh cai": 9000,
    "Canh chua": 10000,
    "Com trang": 5000,
    "Dau hu sot ca": 16000,
    "Ga chien": 25000,
    "Rau muong xao": 8000,
    "Thit kho": 17000,
    "Thit kho trung": 18000,
    "Trung chien": 12000,
}


def resolve_model_path(filename: str) -> str:
    """Tìm file model ở thư mục models/, thư mục gốc repo, hoặc /content/ (Colab)."""
    candidates = [
        BASE_DIR / "models" / filename,
        BASE_DIR / filename,
        Path("/content") / filename,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Không tìm thấy '{filename}'. Hãy đặt file trong '{BASE_DIR}' hoặc '{BASE_DIR / 'models'}'."
    )


detector = YOLO(resolve_model_path("YOLOv8.pt"))
cnn_interpreter = tf.lite.Interpreter(model_path=resolve_model_path("CNNmodel.tflite"))
cnn_interpreter.allocate_tensors()
cnn_input = cnn_interpreter.get_input_details()
cnn_output = cnn_interpreter.get_output_details()


def generate_invoice_id() -> str:
    now_str = datetime.now().strftime("%Y%m%d")
    unique_part = str(uuid.uuid4())[:8].upper()
    return f"INV{now_str}-{unique_part}"


def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history_data: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


def detect_and_classify(img: np.ndarray):
    if img is None:
        return "Vui lòng tải lên hoặc chụp ảnh khay cơm trước.", "0đ", None

    img_h, img_w = img.shape[:2]
    yolo_result = detector(img)
    bboxes = yolo_result[0].boxes.data.cpu().numpy()

    detected_dishes = []
    total_cost = 0

    for box in bboxes:
        xmin, ymin, xmax, ymax, conf, _ = box
        if conf < CONF_THRESHOLD:
            continue

        x1 = max(0, int(xmin))
        y1 = max(0, int(ymin))
        x2 = min(img_w, int(xmax))
        y2 = min(img_h, int(ymax))

        crop_img = img[y1:y2, x1:x2]
        if crop_img.size == 0:
            continue

        crop_resized = cv2.resize(crop_img, INPUT_SIZE)
        input_arr = np.expand_dims(crop_resized.astype(np.float32) / 255.0, axis=0)

        cnn_interpreter.set_tensor(cnn_input[0]["index"], input_arr)
        cnn_interpreter.invoke()
        preds = cnn_interpreter.get_tensor(cnn_output[0]["index"])

        dish_idx = int(np.argmax(preds))
        dish_name = FOOD_LABELS[dish_idx]
        detected_dishes.append(dish_name)
        total_cost += PRICE_LIST.get(dish_name, 0)

    if not detected_dishes:
        return "Không nhận diện được món nào!", "0đ", None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    invoice_id = generate_invoice_id()
    detail_lines = [f"- {name}: {PRICE_LIST[name]:,}đ" for name in detected_dishes]
    summary = "\n".join(detail_lines)
    total_str = f"{total_cost:,}đ"

    invoice_text = (
        f"HÓA ĐƠN: {invoice_id}\n"
        f"Thời gian: {now}\n"
        f"---------------------------\n"
        f"{summary}\n"
        f"---------------------------\n"
        f"TỔNG CỘNG: {total_str}\n"
    )

    invoice_path = INVOICES_DIR / f"{invoice_id}.txt"
    invoice_path.write_text(invoice_text, encoding="utf-8")

    history_data = load_history()
    history_data.append(
        {
            "id": invoice_id,
            "time": now,
            "items": detected_dishes,
            "total": total_cost,
            "file": str(invoice_path),
        }
    )
    save_history(history_data)

    return summary, total_str, str(invoice_path)


def capture_from_camera(camera_url: str):
    url = (camera_url or "").strip()
    if not url:
        return None, "Vui lòng nhập URL/IP camera (RTSP hoặc HTTP)."

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None, "Không thể kết nối tới camera. Kiểm tra lại URL/IP."

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Không lấy được khung hình từ camera."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, "Đã chụp ảnh từ camera thành công."


def view_history() -> str:
    history_data = load_history()
    if not history_data:
        return "Chưa có hóa đơn nào."

    last_5 = history_data[-5:]
    lines = [
        f"{h['id']} | {h['time']} | {', '.join(h['items'])} | {h['total']:,}đ"
        for h in last_5
    ]
    return "\n".join(lines)


def query_invoice_by_id(invoice_id: str) -> str:
    target_id = (invoice_id or "").strip().upper()
    if not target_id:
        return "Vui lòng nhập mã hóa đơn."

    history_data = load_history()
    if not history_data:
        return "Không có dữ liệu lịch sử."

    for record in history_data:
        if record["id"] == target_id:
            invoice_path = Path(record["file"])
            if invoice_path.exists():
                return invoice_path.read_text(encoding="utf-8")
            return f"Tìm thấy mã {target_id} nhưng file hóa đơn không còn tồn tại."

    return f"Không tìm thấy hóa đơn với mã: {target_id}"


with gr.Blocks(theme=gr.themes.Soft()) as app:
    with gr.Row():
        gr.Markdown("<h2>Hệ thống nhận diện và tính tiền khay cơm</h2>")

    with gr.Row():
        inp_img = gr.Image(type="numpy", label="Ảnh khay cơm")
        with gr.Column():
            out_dishes = gr.Textbox(label="Danh sách món & giá", lines=6, interactive=False)
            out_sum = gr.Textbox(label="Tổng tiền", interactive=False)
            invoice_file = gr.File(label="Tải hóa đơn", interactive=False)

    with gr.Row():
        run_btn = gr.Button("Nhận diện và xuất hóa đơn")
        view_history_btn = gr.Button("Xem 5 hóa đơn gần nhất")

    with gr.Row():
        cam_input = gr.Textbox(label="Nhập URL/IP camera (RTSP/HTTP)")
        cam_btn = gr.Button("Chụp ảnh từ camera")
        cam_status = gr.Textbox(label="Trạng thái kết nối", interactive=False)

    captured_img = gr.Image(label="Ảnh từ camera", type="numpy")

    with gr.Row():
        invoice_query = gr.Textbox(label="Nhập mã hóa đơn để tra cứu")
        query_btn = gr.Button("Tra cứu hóa đơn")

    history_output = gr.Textbox(label="Kết quả lịch sử hoặc tra cứu", lines=10, interactive=False)

    run_btn.click(fn=detect_and_classify, inputs=inp_img, outputs=[out_dishes, out_sum, invoice_file])
    view_history_btn.click(fn=view_history, outputs=history_output)
    cam_btn.click(fn=capture_from_camera, inputs=cam_input, outputs=[captured_img, cam_status])
    query_btn.click(fn=query_invoice_by_id, inputs=invoice_query, outputs=history_output)

if __name__ == "__main__":
    app.launch()
