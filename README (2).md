# CUOI_KI_XU_LI_ANH_SO — Conveyor Product Counter (CV + GUI)

Ứng dụng đếm số lượng sản phẩm trên băng chuyền từ:

- **Video/Webcam**: đếm sản phẩm đi qua **một vạch đếm** (line-crossing).
- **Ảnh rời (folder ảnh)**: đếm **số blob trong mỗi ảnh** (in-frame blob counting).

- GUI: Tkinter
- Xử lý ảnh: OpenCV + NumPy
- Kỹ thuật chính: segmentation (threshold hoặc background subtraction) + morphology + connected components/contours + centroid tracking + line-crossing counting.

## 1) Cài đặt

### Yêu cầu
- Python 3.10+ (khuyến nghị)
- Windows (bạn đang dùng) / macOS / Linux đều chạy được

### Cài dependencies
Nếu bạn đã có `requirements.txt` ở root workspace thì vẫn dùng được. Project này kèm một file requirements riêng.

- Cài trong môi trường của workspace:
  - `pip install -r CUOI_KI_XU_LI_ANH_SO/requirements.txt`

## 2) Chạy ứng dụng

- Chạy GUI:
  - `python CUOI_KI_XU_LI_ANH_SO/run_app.py`

Trong GUI:
1. Chọn nguồn (Video / Webcam / Images).
2. Bấm **Open** để mở capture.
3. (Khuyến nghị) Bấm **Select ROI** để khoanh vùng băng chuyền.

Tuỳ bạn chọn **Count mode**:

- `line`: Bấm **Select Line** để chọn 2 điểm tạo vạch đếm → **Start** để chạy đếm.
- `blob`: Không cần chọn line → **Start** để xem “In-frame” (số sản phẩm trong ảnh/khung hiện tại).

Kết quả overlay và mask debug có thể hiển thị trong GUI. Bạn cũng có thể lưu config ra JSON.

## 3) Ý tưởng thuật toán (tóm tắt)

- Tạo mask foreground:
  - Cách 1: Background subtractor (MOG2) → threshold → morphology.
  - Cách 2: Blur → threshold (Otsu/manual) → morphology.
- Tìm blobs sản phẩm: contours/connected components và lọc theo diện tích.
- Tracking: gán centroid theo nearest-neighbor (đủ cho băng chuyền một hướng).
- Đếm: một track **chỉ được cộng 1 lần** khi centroid **cắt qua vạch**.

Gợi ý cho nền tối (video/ảnh):

- Nếu vật thể **sáng trên nền tối**: dùng `seg_mode=threshold` và bật **Invert threshold** nếu thấy mask bị “ngược”.
- Nếu video có nhiễu ánh sáng/độ sáng thay đổi: thử `seg_mode=bgsub` (MOG2) trước.

## 4) Test nhanh

- Chạy unit tests (không cần pytest):
  - Nếu bạn đang đứng trong folder `CUOI_KI_XU_LI_ANH_SO/`:
    - `python -m unittest discover -s tests -v`
  - Nếu bạn chạy từ **workspace root**:
    - `python -m unittest discover -s CUOI_KI_XU_LI_ANH_SO/tests -t CUOI_KI_XU_LI_ANH_SO -v`

## 5) Thư mục

- `conveyor_counter/`: mã nguồn chính
- `outputs/`: file output (log, ảnh overlay nếu bật lưu)
- `assets/`: để bạn bỏ video demo vào đây (tuỳ chọn)

## 6) Gợi ý demo/báo cáo

- Quay 2–3 video ngắn (15–30s) với các điều kiện ánh sáng khác nhau.
- Báo cáo: mô tả pipeline, tham số, bảng so sánh sai số đếm.

## 7) Dataset gợi ý (có thể dùng ngay)

Project hiện tại đếm bằng **CV truyền thống** (foreground mask → blob → tracking → line-crossing), nên dataset tốt nhất là **video băng chuyền camera cố định**. Nếu bạn chưa có video riêng, các nguồn công khai dưới đây phù hợp để demo và viết báo cáo (mình đã kiểm tra link + license hiển thị trên trang).

### A) Conveyor Detection Dataset (Kaggle)

Link: https://www.kaggle.com/datasets/garipovroma/conveyor-detection-dataset

- Dung lượng: ~182 MB
- Cấu trúc: `dht_images/`, `dht_data/`, `training/`
- License: **CC BY-NC-SA 4.0** (phi thương mại)

### B) Conveyor Belt Crossing Detection Dataset (Kaggle)

Link: https://www.kaggle.com/datasets/hanyv10086/conveyor-belt-crossing-detection-dataset

- Dung lượng: ~78 MB
- Cấu trúc: `images/` + `labels/` (thường là nhãn kiểu YOLO)
- License: **CC BY 4.0**
- Ghi chú: dataset thiên về “hành vi qua băng chuyền” (an toàn lao động). Dù không phải đếm sản phẩm, nó có bối cảnh băng chuyền rõ để test ROI/line/segmentation.

### C) Coal Conveyor Belt Anomaly & Foreign Object Dataset (Kaggle)

Link: https://www.kaggle.com/datasets/hanyv10086/coal-conveyor-belt-anomaly-and-foreign-object-dataset

- Dung lượng: ~363 MB
- Cấu trúc: `train/`, `test/` và các lớp (Normal/Oversized/Rock Bolts)
- License: **CC BY 4.0**
- Ghi chú: dùng như “vật thể chạy trên băng” (foreign object) để demo phát hiện/đếm vật thể.

### D) Roboflow Universe (nhiều dataset băng chuyền)

Trang tổng hợp: https://universe.roboflow.com/browse/manufacturing/conveyor-belt

- Ưu điểm: thường export được nhiều format (YOLO/COCO/VOC…)
- Lưu ý: mỗi dataset có license riêng, nhớ xem phần license trước khi dùng.

### E) Multiple Lego Tracking Dataset (Kaggle) — có video băng chuyền

Link: https://www.kaggle.com/datasets/hbahruz/multiple-lego-tracking-dataset

- Nội dung: tác giả **quay 12 video** smartphone về băng chuyền chở LEGO (nhiều góc nhìn: top/front/diagonal), có annotation theo frame.
- Tags trên Kaggle có **Video**.
- License: **Other (specified in description)** (nhớ đọc kỹ phần mô tả license trên Kaggle trước khi dùng trong báo cáo).

Ghi chú dùng với app của mình:

- Nếu dataset có file video (vd. `.mp4`): chọn **Source = Video** và trỏ thẳng tới file.
- Nếu dataset chủ yếu là **video frames** (ảnh rời): chọn **Source = Images** và trỏ tới folder ảnh.

## 8) Bỏ dữ liệu vào project

- Video/ảnh demo: đặt vào `CUOI_KI_XU_LI_ANH_SO/assets/`
- App **không cần annotation** để chạy (vì đếm theo blob). Annotation chỉ cần khi bạn muốn chấm điểm/đánh giá hoặc làm bản YOLO.
