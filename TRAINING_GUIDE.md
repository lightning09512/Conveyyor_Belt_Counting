# Hướng dẫn Training YOLOv8 cho Conveyor Belt Counter

Tài liệu này hướng dẫn bạn cách thu thập dữ liệu từ video băng chuyền hiện tại, gán nhãn (label), và train một model YOLOv8 để app có thể "hiểu" sản phẩm tốt hơn thay vì chỉ dựa vào phân loại màu sắc cơ bản.

---

## Bước 1: Trích xuất ảnh từ Video (Data Collection)

Bạn cần trích xuất các khung hình (frames) từ video băng chuyền để làm dữ liệu training.

Chạy lệnh sau:
```powershell
python tools/prepare_dataset.py --video assets/7913372582276.mp4 --interval 15 --out_dir datasets/images
```

* Tham số `--interval 15` nghĩa là cứ 15 frame video sẽ lấy 1 ảnh (tránh việc lấy các ảnh quá giống nhau).
* Ảnh sẽ được lưu vào thư mục `datasets/images`.

> [!TIP]
> Nếu bạn đã cấu hình khung ROI (Region of Interest) trong app và lưu file `conveyor_config.json`, bạn có thể truyền thêm `--config conveyor_config.json` để script tự động cắt ảnh theo khung ROI. Việc này giúp model tập trung vào khu vực băng chuyền.

---

## Bước 2: Gán nhãn (Labeling)

Bạn cần vẽ khung bao (bounding box) cho các sản phẩm trong ảnh và gán nhãn cho chúng. 

Công cụ khuyên dùng: **Roboflow** (web-based) hoặc **LabelImg** (offline).

### Nếu dùng Roboflow:
1. Tạo tài khoản và Project mới (Object Detection).
2. Upload toàn bộ ảnh trong thư mục `datasets/images` lên Roboflow.
3. Tạo các Class (tên nhãn). 
   > [!IMPORTANT]
   > Nếu bạn muốn app tiếp tục sử dụng logic đếm theo màu (như cũ), hãy tạo các class có tên tiếng Anh chuẩn của màu: `Red`, `Yellow`, `Green`, `Blue`.
   > Nếu bạn muốn đếm theo loại sản phẩm, có thể đặt tên `box`, `bottle`, v.v. (khi đó trong app sẽ hiển thị ở mục "UNKNOWN" hoặc cần sửa nhẹ UI để hiển thị tên mới).
4. Vẽ bounding box cho từng ảnh.
5. Export dataset với định dạng **YOLOv8**.

### Nếu dùng LabelImg:
1. Cài đặt: `pip install labelImg`
2. Mở terminal gõ `labelImg`
3. Chọn thư mục ảnh `datasets/images`.
4. Đổi format lưu sang `YOLO`.
5. Tạo thư mục `datasets/labels` và trỏ "Change Save Dir" về đó.
6. Vẽ box và gán nhãn.

**Cấu trúc thư mục dataset chuẩn YOLOv8 (nếu tự làm):**
```text
datasets/
  ├── images/
  │   ├── train/ (70% số ảnh)
  │   └── val/   (30% số ảnh)
  └── labels/
      ├── train/ (70% file txt)
      └── val/   (30% file txt)
```
Cần tạo thêm file `data.yaml` có nội dung như sau:
```yaml
path: ../datasets  # thư mục gốc chứa dataset
train: images/train
val: images/val

names:
  0: Red
  1: Yellow
  2: Green
  3: Blue
```

---

## Bước 3: Cài đặt thư viện Training

Cài đặt package `ultralytics` (YOLOv8):

```powershell
pip install ultralytics
```

---

## Bước 4: Training

Sau khi tải dataset từ Roboflow về (bạn sẽ có file `data.yaml` bên trong thư mục dataset), hãy chạy script train:

```powershell
python tools/train_yolo.py --data "đường_dẫn_tới_thư_mục_dataset/data.yaml" --epochs 50 --imgsz 640
```

> [!NOTE]
> Quá trình train có thể mất từ vài chục phút đến vài giờ tùy vào việc máy bạn có GPU mạnh (như NVIDIA RTX) hay không.

Sau khi train xong, model tốt nhất sẽ được lưu tại: `runs/train/conveyor_model/weights/best.pt`

---

## Bước 5: Sử dụng Model trong App

1. Mở app: `python run_app.py`
2. Mở Sidebar (CONTROLS & PARAMETERS).
3. Tại phần **DETECTION MODE**, chọn `Mode: yolo`.
4. Bấm `Browse` và chọn file `best.pt` vừa được tạo ra ở bước 4.
5. Bấm `Load YOLO Model`.
6. Tinh chỉnh `Confidence` (ngưỡng tự tin, thường từ 0.3 đến 0.6).

Bây giờ hệ thống đếm sản phẩm sẽ sử dụng AI YOLOv8 thay vì xử lý ảnh truyền thống!
