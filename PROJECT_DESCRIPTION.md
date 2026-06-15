# Conveyor Belt Product Counter
## Hệ thống đếm sản phẩm trên băng chuyền sử dụng Xử lý ảnh

---

## 1. Tổng quan dự án

**Conveyor Belt Product Counter** là một ứng dụng desktop thời gian thực (real-time) sử dụng các kỹ thuật **Xử lý ảnh truyền thống (Traditional Computer Vision)** — không dùng Deep Learning — để phát hiện, theo dõi và đếm các sản phẩm/vật thể di chuyển trên băng chuyền công nghiệp.

Ứng dụng được xây dựng hoàn toàn bằng **Python**, sử dụng thư viện **OpenCV** cho phần xử lý ảnh và **CustomTkinter** cho giao diện đồ họa người dùng (GUI).

### Đặc điểm chính
- Phát hiện và đếm vật thể theo thời gian thực
- Phân loại màu sắc sản phẩm tự động (Đỏ, Vàng, Xanh lá, Xanh dương)
- Tách các vật thể dính nhau bằng thuật toán xâm thực hình thái (Erosion-based Splitting)
- Theo dõi vật thể (Object Tracking) qua nhiều khung hình liên tiếp
- Đếm chính xác khi vật thể vượt qua vạch đếm (Line Crossing)
- Giao diện GUI hiện đại với khả năng cấu hình linh hoạt

---

## 2. Mô hình và thuật toán sử dụng

### 2.1. Trừ nền — Background Subtraction (MOG2)

**Mô hình:** `cv2.createBackgroundSubtractorMOG2`

Đây là bước đầu tiên và quan trọng nhất trong pipeline. Thuật toán **MOG2 (Mixture of Gaussians 2)** được sử dụng để tách vật thể di chuyển (foreground) ra khỏi nền tĩnh (background) của băng chuyền.

**Nguyên lý hoạt động:**
- MOG2 mô hình hóa mỗi pixel nền bằng một **hỗn hợp các phân phối Gaussian** (Gaussian Mixture Model — GMM)
- Mỗi khung hình mới, thuật toán so sánh giá trị pixel với các phân phối đã học:
  - Nếu pixel **khớp** với một Gaussian → pixel thuộc nền (background)
  - Nếu pixel **không khớp** → pixel thuộc tiền cảnh (foreground = vật thể)
- Mô hình được cập nhật liên tục với `learningRate`, cho phép thích ứng với thay đổi ánh sáng dần dần
- Hỗ trợ **phát hiện bóng đổ** (shadow detection) để loại bỏ nhiễu từ bóng vật thể

**Tham số cấu hình:**

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|-------|
| `bg_history` | 300 | Số khung hình để xây dựng mô hình nền |
| `bg_var_threshold` | 36 | Ngưỡng phương sai để phân loại foreground/background |
| `bg_detect_shadows` | True | Bật phát hiện bóng đổ (shadow value = 127) |

**Đầu ra:** Binary mask (ảnh nhị phân) — pixel trắng (255) = vật thể, pixel đen (0) = nền.

---

### 2.2. Tiền xử lý hình thái — Morphological Operations

Sau khi có binary mask từ MOG2, các phép toán hình thái học được áp dụng để **làm sạch** mask:

**Các phép toán sử dụng:**

1. **Phép đóng (Morphological Close)** — `cv2.MORPH_CLOSE`
   - Sử dụng phần tử cấu trúc (structuring element) hình vuông kích thước `kernel_size × kernel_size`
   - **Mục đích:** Lấp đầy các lỗ nhỏ bên trong vật thể (fill gaps)
   - **Nguyên lý:** Dilation → Erosion

2. **Phép mở (Morphological Open)** — `cv2.MORPH_OPEN`
   - **Mục đích:** Loại bỏ nhiễu nhỏ (noise removal)
   - **Nguyên lý:** Erosion → Dilation

**Tham số:**

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|-------|
| `morph_kernel` | 7 | Kích thước kernel (luôn là số lẻ) |
| `morph_iters` | 3 | Số lần lặp cho mỗi phép toán |

---

### 2.3. Phát hiện vật thể — Contour Detection

**Thuật toán:** `cv2.findContours` với chế độ `RETR_EXTERNAL`

Sau khi mask đã được làm sạch, thuật toán tìm đường viền (contour) được áp dụng để xác định vị trí và kích thước từng vật thể.

**Quy trình:**
1. Tìm tất cả contour ngoài cùng (`RETR_EXTERNAL`) trên binary mask
2. Với mỗi contour:
   - Tính diện tích (`cv2.contourArea`) → lọc theo `min_area` / `max_area`
   - Tính bounding box (`cv2.boundingRect`) → tọa độ `(x, y, w, h)`
   - Tính centroid (tâm) = `(x + w/2, y + h/2)` → dùng cho tracking

**Tham số lọc:**

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|-------|
| `min_area` | 600 px² | Diện tích tối thiểu để coi là vật thể |
| `max_area` | 60000 px² | Diện tích tối đa cho một vật thể đơn lẻ |

---

### 2.4. Tách vật thể dính — Erosion-based Splitting

Khi các vật thể nằm sát nhau trên băng chuyền, chúng có thể bị nhận nhầm thành **một blob lớn**. Thuật toán tách dựa trên xâm thực (erosion) được sử dụng:

**Nguyên lý:**
1. Nếu diện tích contour > `max_area × 1.5` → nghi ngờ là nhiều vật thể dính
2. Áp dụng `cv2.erode` với cường độ tăng dần (iterations = 2, 3, 4, 5)
3. Phần mỏng nối giữa 2 vật thể bị xâm thực mạnh hơn → bị đứt ra
4. Tìm lại contour trên mask đã erode → nếu có ≥ 2 contour hợp lệ → tách thành công
5. Dilate ngược lại để khôi phục kích thước gốc của từng vật thể

**Ưu điểm so với Watershed:**
- Nhanh hơn rất nhiều (~1ms vs ~20ms)
- Ổn định hơn, ít bị lỗi phân vùng sai

---

### 2.5. Phân loại màu — HSV Color Classification

**Không gian màu:** HSV (Hue — Saturation — Value)

Mỗi vật thể được phân loại màu bằng cách tính **giá trị trung bình HSV** trong vùng mask của nó:

```
cv2.mean(hsv_roi, mask=roi_mask) → (H, S, V)
```

**Bảng phân loại:**

| Màu | Phạm vi Hue (H) | Điều kiện bổ sung |
|-----|------------------|-------------------|
| Đỏ (Red) | H < 12 hoặc H > 165 | S ≥ 50, V ≥ 50 |
| Vàng (Yellow) | 15 < H < 35 | S ≥ 50, V ≥ 50 |
| Xanh lá (Green) | 35 ≤ H < 85 | S ≥ 50, V ≥ 50 |
| Xanh dương (Blue) | 85 ≤ H ≤ 130 | S ≥ 50, V ≥ 50 |
| Không xác định | Các trường hợp còn lại | S < 50 hoặc V < 50 |

---

### 2.6. Theo dõi vật thể — Centroid Tracker

**Thuật toán:** Centroid-based Nearest Neighbor Tracking

Thuật toán theo dõi đơn giản nhưng hiệu quả cho bài toán băng chuyền, nơi vật thể di chuyển đều và ít giao nhau.

**Nguyên lý hoạt động:**
1. Mỗi vật thể được gán một **Track ID** duy nhất
2. Ở mỗi khung hình mới:
   - Tính **ma trận khoảng cách** giữa tất cả track hiện tại và detection mới (Euclidean distance)
   - Áp dụng **phạt khoảng cách** (+200px) nếu màu sắc không khớp → tránh hoán đổi ID giữa vật thể khác màu
   - **Greedy Assignment:** ghép cặp (track, detection) theo thứ tự khoảng cách tăng dần
   - Detection không ghép được → tạo Track mới
   - Track không ghép được → tăng `missing` counter → xóa nếu vượt `max_missing_frames`

**Tham số:**

| Tham số | Giá trị mặc định | Mô tả |
|---------|-------------------|-------|
| `max_match_distance` | 120.0 px | Khoảng cách tối đa để ghép track-detection |
| `max_missing_frames` | 15 frames | Số frame tối đa track bị mất trước khi xóa |

---

### 2.7. Đếm vượt vạch — Line Crossing Detection

**Thuật toán:** Cross Product Sign Change

Phương pháp hình học để xác định vật thể đã vượt qua vạch đếm:

**Nguyên lý:**
1. Vạch đếm được biểu diễn bằng 2 điểm: `(x1, y1) — (x2, y2)` → đường thẳng có hướng
2. Với mỗi track, tính giá trị `side_value` (tích chéo — cross product):
   ```
   side_value(P) = (x2 - x1) × (Py - y1) - (y2 - y1) × (Px - x1)
   ```
   - `side_value > 0` → điểm P nằm bên trái đường thẳng
   - `side_value < 0` → điểm P nằm bên phải đường thẳng
3. So sánh vị trí centroid ở **frame trước** (`prev`) và **frame hiện tại** (`cur`):
   - Nếu `side_value(prev) × side_value(cur) < 0` → **đã cắt qua vạch** (đổi dấu nghiêm ngặt)
4. Mỗi track chỉ được đếm **đúng một lần** (flag `counted = True`)

---

## 3. Pipeline tổng thể

```
┌─────────────────┐
│  Video / Webcam  │
│  / Ảnh tĩnh     │
└────────┬────────┘
         │ frame (BGR)
         ▼
┌─────────────────┐
│  Crop ROI       │  ← Vùng quan tâm (tùy chọn)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MOG2 Background│  ← Trừ nền → binary mask
│  Subtraction    │
└────────┬────────┘
         │ mask (0/255)
         ▼
┌─────────────────┐
│  Morphological  │  ← CLOSE + OPEN → làm sạch mask
│  Postprocessing │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Contour        │  ← Tìm đường viền → lọc theo diện tích
│  Detection      │
│  + Erosion Split│  ← Tách blob dính bằng erosion
│  + Color Class. │  ← Phân loại màu HSV
└────────┬────────┘
         │ list[Detection]
         ▼
┌─────────────────┐
│  Centroid       │  ← Ghép detection → track ID
│  Tracker        │
└────────┬────────┘
         │ dict[track_id → Track]
         ▼
┌─────────────────┐
│  Line Crossing  │  ← Kiểm tra cắt vạch đếm
│  Counter        │  ← Cập nhật tổng số + theo màu
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │  ← Vẽ bbox, ID, vạch đếm, thống kê
│  + GUI Display  │
└─────────────────┘
```

---

## 4. Cấu trúc mã nguồn

```
Conveyyor_Belt_Counting/
├── run_app.py                    # Entry point — khởi chạy ứng dụng
├── requirements.txt              # Thư viện phụ thuộc
├── conveyor_config.example.json  # File cấu hình mẫu
├── assets/                       # Video mẫu để test
│   ├── color box.mp4
│   ├── 1.mp4, 2.mp4, 3.mp4
│   └── ...
└── conveyor_counter/             # Package chính
    ├── __init__.py
    ├── app.py                    # GUI (CustomTkinter) + vòng lặp xử lý chính
    ├── config.py                 # Cấu trúc cấu hình (AppConfig, ROI, Line)
    ├── vision.py                 # Xử lý ảnh: MOG2, morphology, contour, color
    ├── tracker.py                # Centroid Tracker + Line Crossing Counter
    └── geometry.py               # Hình học: Point, Line2D, crossed_line
```

---

## 5. Công nghệ và thư viện sử dụng

| Thư viện | Phiên bản | Vai trò |
|----------|-----------|---------|
| **OpenCV** (`opencv-python`) | ≥ 4.5 | Xử lý ảnh, video, background subtraction, morphology |
| **NumPy** | ≥ 1.20 | Tính toán ma trận, khoảng cách Euclidean |
| **Pillow** (PIL) | ≥ 8.0 | Chuyển đổi ảnh OpenCV → Tkinter để hiển thị |
| **CustomTkinter** | ≥ 5.0 | Framework GUI hiện đại (dark mode) |
| **Python** | ≥ 3.10 | Ngôn ngữ lập trình chính |

---

## 6. Chế độ hoạt động

### 6.1. Chế độ đếm theo vạch (Line Counting Mode)
- Phù hợp với **video** và **webcam**
- Vật thể được đếm khi centroid cắt qua vạch đếm do người dùng đặt
- Mỗi vật thể chỉ được đếm **1 lần** nhờ hệ thống tracking

### 6.2. Chế độ đếm blob (Blob Counting Mode)
- Phù hợp với **ảnh tĩnh** hoặc đếm tức thì
- Đếm số lượng vật thể hiện có trong khung hình tại thời điểm hiện tại
- Không sử dụng tracking

### 6.3. Chế độ phát hiện đối tượng (Detection Modes)
- **Traditional CV**: Chế độ mặc định, sử dụng Background Subtraction (MOG2) và xử lý hình thái học để tìm viền (contour). Không cần training, rất nhanh, nhưng nhạy cảm với ánh sáng và nhiễu nền.
- **YOLOv8**: Chế độ nâng cao sử dụng Deep Learning. Cần cung cấp file model `.pt` đã được huấn luyện. Cho phép phát hiện đối tượng chính xác cao hơn, kể cả khi các đối tượng bị che khuất một phần hoặc dính sát vào nhau. Tích hợp sẵn bộ công cụ huấn luyện (trong thư mục `tools/`).

### 6.4. Nguồn đầu vào hỗ trợ
- **Video file** (MP4, AVI, MKV...)
- **Webcam** (USB camera, camera tích hợp)
- **Thư mục ảnh** (xử lý từng ảnh một)

---

## 7. Hướng dẫn sử dụng nhanh

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy ứng dụng
python run_app.py
```

**Trên giao diện:**
1. Nhấp **"Open"** → chọn video hoặc webcam
2. *(Tùy chọn)* Nhấp **"Set ROI"** → vẽ vùng quan tâm trên ảnh
3. Nhấp **"Set Line"** → vẽ vạch đếm bằng cách click 2 điểm
4. Nhấp **"Start"** → bắt đầu xử lý và đếm

---

## 8. Hiệu năng

| Chỉ số | Giá trị |
|--------|---------|
| Tốc độ xử lý trung bình | ~24 ms/frame |
| FPS hiệu dụng | ~41 FPS |
| Độ phân giải hỗ trợ | Tùy ý (tự động resize) |
| Yêu cầu GPU | **Không** — chạy hoàn toàn trên CPU |

---

*Tài liệu này được tạo tự động. Cập nhật lần cuối: 15/06/2026.*
