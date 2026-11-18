# Hệ thống Nhận diện Khay cơm & Tính tiền Tự động (Smart Canteen)

Dự án cá nhân ứng dụng Thị giác máy tính (Computer Vision) để tự động nhận diện các món ăn Việt Nam trên khay cơm và xuất hóa đơn thanh toán theo thời gian thực qua giao diện web.

---

## Kiến trúc hệ thống (Two-Stage Pipeline)

1. **Phát hiện vùng món ăn (Object Detection):** Sử dụng **YOLOv8** (`YOLOv8.pt`) để khoanh vùng từng món ăn trên ảnh khay cơm (lọc ngưỡng tin cậy `conf >= 0.3`).
2. **Phân loại món ăn (Image Classification):** Cắt từng vùng bounding box, resize về kích thước `224×224`, chuẩn hóa pixel `[0, 1]` và đưa qua mô hình **CNN chuyển đổi sang TensorFlow Lite** (`CNNmodel.tflite`) để phân loại vào 10 lớp món ăn.
3. **Tính tiền & Quản lý hóa đơn:** Đối chiếu bảng giá, tính tổng hóa đơn, xuất file `.txt` (`invoices/`) và lưu lịch sử giao dịch (`history.json`) để tra cứu theo mã.
4. **Giao diện & Camera:** Xây dựng web UI bằng **Gradio**, hỗ trợ tải ảnh trực tiếp hoặc chụp khung hình từ camera IP (`RTSP/HTTP`) thông qua **OpenCV**.

---

## Danh sách 10 món ăn hỗ trợ

| Món ăn | Giá mặc định | Món ăn | Giá mặc định |
| :--- | :---: | :--- | :---: |
| Cá hú kho | 22,000đ | Gà chiên | 25,000đ |
| Canh cải | 9,000đ | Rau muống xào | 8,000đ |
| Canh chua | 10,000đ | Thịt kho | 17,000đ |
| Cơm trắng | 5,000đ | Thịt kho trứng | 18,000đ |
| Đậu hũ sốt cà | 16,000đ | Trứng chiên | 12,000đ |

---

## Cấu trúc thư mục

```text
├── app.py              # Mã nguồn ứng dụng Gradio & pipeline YOLOv8 + TFLite
├── YOLOv8.pt           # Trọng số mô hình phát hiện món ăn
├── CNNmodel.tflite     # Mô hình CNN phân loại món ăn (TensorFlow Lite)
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md
```

---

## Hướng dẫn cài đặt & chạy thử

### Cách 1: Chạy trên máy cá nhân (Local)

```bash
# 1. Clone repository
git clone https://github.com/NhanRTX/vietnamese-food-tray-recognition.git
cd vietnamese-food-tray-recognition

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Khởi chạy ứng dụng
python app.py
```

Sau khi chạy, mở đường dẫn local hiển thị trên terminal (mặc định `http://127.0.0.1:7860`) để sử dụng.

### Cách 2: Chạy trên Google Colab

```bash
!git clone https://github.com/NhanRTX/vietnamese-food-tray-recognition.git
%cd vietnamese-food-tray-recognition
!pip install -r requirements.txt
!python app.py
```

---

## Demo giao diện

> <img width="1695" height="905" alt="image" src="https://github.com/user-attachments/assets/94c28962-0560-44e5-a681-34385e4ee60b" />


---

## Tác giả

- **Nguyễn Thành Nhân** — [GitHub @NhanRTX](https://github.com/NhanRTX)
