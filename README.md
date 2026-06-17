# Conveyor Belt Product Counter

Hệ thống đếm sản phẩm trên băng chuyền bằng xử lý ảnh (OpenCV) và tùy chọn YOLOv8.

## Cài đặt & chạy

```powershell
cd "d:\hoc\Xu Ly Anh\Conveyyor_Belt_Counting"
pip install -r requirements.txt
python run_app.py
```

**Lưu ý:** Video/ảnh demo (`.mp4`, `.jpg`…) không được commit vào git. Đặt file video vào thư mục `assets/` trước khi chạy, hoặc dùng webcam.

**Config mẫu:** `conveyor_config.example.json` hoặc `assets/7913372582276_config.json` (cần file `assets/7913372582276.mp4` tương ứng).

**Huấn luyện YOLOv8:** xem `TRAINING_GUIDE.md`.

**Chạy tests:**

```powershell
python -m unittest discover -s tests
```

---

# 🎓 TÀI LIỆU ÔN TẬP — Báo Cáo Project Đếm Sản Phẩm Băng Chuyền

> Mục tiêu: Nắm vững tất cả kiến thức cần thiết để trình bày và trả lời câu hỏi của thầy.

---

## PHẦN 1: ĐƯỜNG ỐNG XỬ LÝ — "Luồng chạy từ đầu đến cuối"

> [!TIP]
> Đây là phần QUAN TRỌNG NHẤT. Thầy sẽ hỏi: "Từ khi nhận khung hình video cho đến khi đếm được sản phẩm, hệ thống làm qua mấy bước?"

### Trả lời: 7 bước chính

```
Khung hình (BGR) → ❶ Cắt vùng quan tâm → ❷ Phân vùng ảnh → ❸ Xử lý hình thái
                 → ❹ Phát hiện đường viền → ❺ Phân loại màu sắc
                 → ❻ Theo dõi đối tượng → ❼ Đếm qua vạch
```

#### ❶ Cắt vùng quan tâm
**Mục đích:** Chỉ xử lý phần ảnh có băng chuyền, bỏ qua viền thừa (tường, dây điện, nền ngoài khung hình) → giảm nhiễu, tập trung vào vùng cần đếm và chạy nhanh hơn.

- Người dùng vẽ vùng quan tâm trên giao diện.
- Hệ thống chỉ xử lý phần ảnh nằm trong vùng này → **giúp giảm nhiễu, tăng tốc độ xử lý**.
- Mã nguồn: Hàm `crop_roi()` trong tệp `vision.py`.

#### ❷ Phân vùng ảnh — tách sản phẩm khỏi nền
**Mục đích:** Tách sản phẩm (tiền cảnh) ra khỏi nền băng chuyền, tạo ảnh nhị phân (mask) — vùng trắng là vật thể, vùng đen là nền — làm đầu vào cho các bước phát hiện phía sau.

**Có 2 phương pháp:**

| Tiêu chí | Trừ nền (MOG2) | Phân ngưỡng (Otsu hoặc Thủ công) |
|---|---|---|
| **Khi nào dùng** | Video hoặc Camera (nền chuyển động) | Ảnh tĩnh hoặc khi nền hoàn toàn cố định |
| **Cách hoạt động** | Mô hình hóa nền bằng hỗn hợp Gauss (GMM), điểm ảnh khác mô hình → tiền cảnh | Chuyển sang ảnh xám → so sánh với ngưỡng → tạo ảnh nhị phân |
| **Hàm OpenCV** | `cv2.createBackgroundSubtractorMOG2()` | `cv2.threshold()` |
| **Ưu điểm** | Tự thích ứng với những thay đổi về ánh sáng | Đơn giản, tính toán nhanh |
| **Nhược điểm** | Cần vài khung hình đầu để "học" nền | Rất nhạy cảm với sự thay đổi ánh sáng |

**Tham số trừ nền MOG2 cần nhớ:**
- `history = 300` → dùng 300 khung hình gần nhất để xây dựng mô hình nền.
- `varThreshold = 36` → ngưỡng phương sai Mahalanobis để quyết định phân loại điểm ảnh.
- `detectShadows = True` → bật tính năng phát hiện bóng đổ (bóng được gán giá trị 127), sau đó loại bỏ bằng phép phân ngưỡng tại mức 200.

**Phân ngưỡng Otsu:**
- Thuật toán tự tìm ngưỡng tối ưu dựa trên **biểu đồ phân bố cường độ có dạng hai đỉnh** (một đỉnh cho nền, một đỉnh cho đối tượng).
- Công thức: Tìm giá trị ngưỡng sao cho **phương sai liên lớp** đạt cực đại.
- Nói cách khác: Tìm ngưỡng để tách 2 nhóm điểm ảnh (nền và đối tượng) sao cho 2 nhóm **khác biệt nhau nhiều nhất**.

#### ❸ Phép toán hình thái học
**Mục đích:** Làm sạch mask sau bước phân vùng — xóa đốm nhiễu nhỏ, lấp lỗ hổng, làm viền vật thể mượt hơn → giúp bước tìm đường viền và đếm chính xác hơn.

1. **Phép mở** = Phép Co rồi đến Phép Nở: Dùng để loại bỏ các đốm nhiễu nhỏ bên ngoài đối tượng.
2. **Phép đóng** = Phép Nở rồi đến Phép Co: Dùng để lấp đầy các lỗ hổng nhỏ bên trong đối tượng.

**Tham số:** Phần tử cấu trúc (ma trận nhân) hình vuông kích thước 11×11 (mặc định), số lần lặp = 3.

**Cách giải thích cho thầy:**
> "Sau khi phân ngưỡng, ảnh nhị phân thường có nhiễu (như các đốm trắng nhỏ ở nền, hoặc lỗ hổng bên trong sản phẩm). Phép mở giúp xóa đốm nhiễu nhỏ, phép đóng giúp lấp lỗ hổng → kết quả là ảnh nhị phân sạch hơn, từ đó phát hiện đường viền chính xác hơn."

#### ❹ Phát hiện đường viền
**Mục đích:** Tìm từng vật thể riêng lẻ trên mask — xác định vị trí (hình chữ nhật bao quanh), tâm và diện tích của mỗi sản phẩm để đưa sang bước theo dõi và đếm.

- Sử dụng hàm `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
- Tham số `RETR_EXTERNAL`: Chỉ lấy đường viền bên ngoài cùng (bỏ qua các đường viền lồng nhau bên trong).
- Lọc nhiễu theo **diện tích**: `diện_tích_tối_thiểu ≤ diện_tích ≤ diện_tích_tối_đa` (từ 600 đến 60000 điểm ảnh).
- Tính tọa độ tâm đối tượng: `cx = x + chiều_rộng/2, cy = y + chiều_cao/2`.
- Tính hình chữ nhật bao quanh: Hàm `cv2.boundingRect(contour)` → `(x, y, rộng, cao)`.

#### ❺ Phân loại màu sắc — Dùng Không gian màu HSV
**Mục đích:** Gán nhãn màu (Đỏ, Vàng, Xanh lá, Xanh dương) cho từng sản phẩm → đếm riêng theo từng màu trên giao diện, đồng thời giúp phân biệt vật dính nhau nếu chúng khác màu.

**Tại sao dùng HSV thay vì RGB?**
> "Trong hệ màu RGB, cùng một màu đỏ nhưng khi sáng hoặc tối khác nhau sẽ có giá trị R, G, B thay đổi rất lớn. Trong hệ màu HSV, kênh Sắc độ (H) chỉ biểu diễn loại màu sắc → do đó ít bị ảnh hưởng bởi sự thay đổi của ánh sáng."

**3 thành phần của HSV:**
- **H (Sắc độ)**: Biểu diễn loại màu (đỏ, vàng, xanh...) — giá trị từ 0-180 trong thư viện OpenCV.
- **S (Độ bão hòa)**: Biểu diễn màu đậm hay nhạt — từ 0 (trắng xám) đến 255 (màu đậm nhất).
- **V (Độ sáng)**: Biểu diễn sáng hay tối — từ 0 (đen) đến 255 (sáng nhất).

**Bảng phân loại màu (CẦN NHỚ):**

| Màu | Sắc độ H | Điều kiện về độ bão hòa và độ sáng |
|-----|----------|----------------|
| **Đỏ** | H < 12 hoặc H > 165 | S ≥ 50 và V ≥ 50 |
| **Vàng** | 15 < H < 35 | S ≥ 50 và V ≥ 50 |
| **Xanh lá** | 35 ≤ H < 85 | S ≥ 50 và V ≥ 50 |
| **Xanh dương** | 85 ≤ H ≤ 130 | S ≥ 50 và V ≥ 50 |
| **Không xác định** | Các trường hợp còn lại | S < 50 hoặc V < 50 |

> [!NOTE]
> Màu đỏ có 2 khoảng giá trị vì dải sắc độ đỏ nằm ở cả 2 đầu của vòng tròn HSV (0° và 360°).
> Trong OpenCV, giá trị H nằm trong khoảng [0, 180] (do chia đôi góc 360° để vừa với kiểu dữ liệu 8-bit), nên màu đỏ nằm ở `H < 12` VÀ `H > 165`.

#### ❻ Theo dõi đối tượng — Thuật toán theo dõi theo tâm
**Mục đích:** Gán ID cố định cho mỗi sản phẩm qua các khung hình video — biết vật ở frame trước và frame sau là cùng một sản phẩm, tránh đếm trùng hoặc mất đếm khi sản phẩm di chuyển trên băng chuyền.

**Thuật toán: Gán tham lam theo tâm gần nhất**

```
Khung hình N:     Vết1(tâm 100,200)       Vết2(tâm 300,400)      ← Các vết theo dõi cũ
Khung hình N+1:   PhátHiện_A(tâm 105,205) PhátHiện_B(tâm 310,405) ← Các phát hiện mới

Ma trận khoảng cách hình học:
            PhátHiện_A   PhátHiện_B
Vết1         7.07         212.13
Vết2        198.49         11.18

→ Ghép Vết1 ↔ PhátHiện_A (khoảng cách gần nhất: 7.07)
→ Ghép Vết2 ↔ PhátHiện_B (khoảng cách gần nhất: 11.18)
```

**Các bước của thuật toán:**
1. Tính **ma trận khoảng cách** giữa tọa độ tâm của các vết cũ và tâm của các đối tượng vừa phát hiện.
2. **Gán cặp tham lam**: Luôn ưu tiên ghép cặp có khoảng cách nhỏ nhất. Nếu khoảng cách lớn hơn mức cho phép (ngưỡng 60 điểm ảnh) → từ chối ghép cặp.
3. Vết cũ không được gán → tăng bộ đếm "mất tích". Nếu số khung hình mất tích > 15 → **xóa vết cũ**.
4. Phát hiện mới không được gán → **tạo vết theo dõi mới**.

**Phạt khoảng cách theo màu:** Nếu vết theo dõi và phát hiện mới **khác màu nhau** → thuật toán tự động cộng thêm +200 điểm ảnh vào khoảng cách → giúp tránh tình trạng hoán đổi mã định danh (ID) giữa hai vật thể khác màu khi chúng đi ngang qua nhau.

#### ❼ Đếm qua vạch — Phương pháp tích chéo
**Mục đích:** Cộng số đếm đúng **một lần** khi tâm sản phẩm đi qua vạch kẻ do người dùng đặt — đây là bước cuối cùng, biến việc “nhìn thấy sản phẩm” thành “đã đếm được bao nhiêu sản phẩm”.

**Công thức tính vị trí tương đối so với vạch kẻ:**
```
giá_trị_phía_của_điểm_P = (x₂-x₁)(Py-y₁) - (y₂-y₁)(Px-x₁)
```
- Nếu `giá_trị_phía > 0` → tâm P nằm **bên trái** vạch kẻ.
- Nếu `giá_trị_phía < 0` → tâm P nằm **bên phải** vạch kẻ.
- So sánh vị trí tâm ở **khung hình trước** và **khung hình hiện tại**:
  - Nếu **đổi dấu** (một điểm nằm bên trái, một điểm nằm bên phải) → chứng tỏ tâm đối tượng đã **cắt qua vạch** → **tăng số đếm lên 1**.
- Mỗi vết theo dõi chỉ được đếm **1 lần duy nhất** (quản lý qua cờ trạng thái `đã_đếm`).

---

## PHẦN 2: CÁC TÍNH NĂNG NÂNG CAO

### 2A. Tách vật dính cùng màu — Watershed
**Khi nào cần?** Khi nhiều sản phẩm **cùng màu** nằm sát nhau → contour gộp thành một blob lớn.

**Cách hoạt động (trong `vision.py`):**
1. Nếu `diện_tích > max(min_area × 1.8, 3500)` → nghi ngờ cụm vật dính.
2. Dùng **distance transform** tìm tâm từng vật.
3. Áp dụng **Watershed** để tách ranh giới.
4. Nếu tách được ≥ 2 vật hợp lệ → đếm riêng từng vật; nếu không và diện tích > `max_area` → bỏ qua.

### 2B. Phân vùng kết hợp theo màu sắc
> Đây là điểm nhấn quan trọng trong mã nguồn thực tế.

**Quy trình:**
1. Tạo **ảnh nhị phân theo từng màu** riêng biệt (Mặt nạ Đỏ, Mặt nạ Vàng, Mặt nạ Xanh lá, Mặt nạ Xanh dương).
2. Dùng **PHÉP VÀ (AND)** với ảnh nhị phân trừ nền → chỉ giữ lại những vùng VỪA có màu sắc tương ứng VỪA đang chuyển động.
3. Tìm đường viền trên từng ảnh nhị phân màu → biết ngay màu sắc của mỗi vật thể từ sớm.

**Tại sao tốt hơn?** Phương pháp này giúp tách biệt tốt 2 vật dính nhau nếu chúng **có màu khác nhau** (ví dụ: hộp đỏ dính sát hộp xanh → vì chúng ở 2 mặt nạ màu khác biệt nên không bao giờ bị gộp chung).

### 2C. Biến đổi phân thủy — Tách vật cùng màu bị dính
Khi hệ thống gặp một cụm quá lớn và cùng màu:
1. Thực hiện **biến đổi khoảng cách** → tìm khoảng cách từ mỗi điểm ảnh bên trong đối tượng đến biên gần nhất.
2. Tìm các **đỉnh cực đại cục bộ** → các đỉnh này chính là tâm của mỗi vật thể riêng lẻ.
3. Dùng thuật toán **Phân thủy** → tưởng tượng nước tràn ra từ các tâm này → ranh giới gặp nhau giữa các dòng nước chính là ranh giới cắt rời 2 vật.

### 2D. Phát hiện đối tượng bằng học sâu YOLOv8
- Tích hợp mô hình học sâu **YOLOv8** (You Only Look Once).
- Phát hiện toàn bộ đối tượng trong **1 lần chạy mạng nơ-ron** trên ảnh.
- Ưu điểm: Độ chính xác rất cao, vượt trội phương pháp truyền thống, không bị ảnh hưởng bởi ánh sáng môi trường.
- Nhược điểm: Cần có dữ liệu mẫu và thời gian huấn luyện mô hình, tốc độ xử lý chậm hơn nếu máy tính không có card đồ họa rời (GPU).

---

## PHẦN 3: CÂU HỎI THẦY CÓ THỂ HỎI & CÁCH TRẢ LỜI

### ❓ Q1: "Thuật toán trừ nền MOG2 hoạt động như thế nào?"
> "Thưa thầy, MOG2 mô hình hóa từng điểm ảnh của nền tĩnh bằng **hỗn hợp các phân phối Gauss**. Khi có một khung hình mới, nó sẽ lấy giá trị điểm ảnh so sánh với mô hình nền đã học. Nếu điểm ảnh khớp với 1 phân phối Gauss → nó là phần nền. Nếu khác biệt hoàn toàn → nó là tiền cảnh (tức là sản phẩm đang di chuyển). Do mô hình được cập nhật theo thời gian, nó có khả năng tự thích ứng khi ánh sáng thay đổi."

### ❓ Q2: "Tại sao em dùng không gian màu HSV mà không dùng RGB?"
> "Dạ, vì hệ màu RGB không tách biệt rõ thông tin về loại màu sắc và độ sáng. Cùng một màu đỏ nhưng khi ở trong bóng râm sẽ có giá trị RGB khác hẳn khi ở ngoài sáng. Hệ màu HSV tách riêng thành 3 phần: Sắc độ (loại màu), Độ bão hòa và Độ sáng. Kênh Sắc độ (H) gần như không đổi khi ánh sáng thay đổi, giúp phân loại màu sắc ổn định và chính xác hơn rất nhiều."

### ❓ Q3: "Phép mở và Phép đóng khác nhau thế nào?"
> "Thưa thầy, **phép mở** là thực hiện phép co rồi mới đến phép nở. Nó làm co nhỏ đối tượng nên các đốm nhiễu nhỏ bên ngoài sẽ biến mất, sau đó nở ra để khôi phục kích thước. Công dụng là **xóa nhiễu nhỏ**. Ngược lại, **phép đóng** là nở rồi mới co. Khi nở ra, các lỗ hổng bên trong đối tượng bị lấp đầy, sau đó co lại giữ nguyên kích thước. Công dụng là **lấp lỗ hổng bên trong đối tượng**."

### ❓ Q4: "Thuật toán theo dõi đối tượng của em có nhược điểm gì?"
> "Thuật toán gán khoảng cách gần nhất dễ bị nhầm lẫn khi 2 sản phẩm đi quá sát nhau hoặc đi đè lên nhau, gây ra hiện tượng tráo đổi mã định danh (ID). Để khắc phục một phần, em đã thêm kỹ thuật **phạt khoảng cách theo màu sắc**: Nếu 2 vật thể có màu khác nhau, thuật toán tự động cộng thêm khoảng cách phạt rất lớn, giúp chúng không bao giờ bị ghép nhầm ID. Tuy nhiên, nếu là 2 vật cùng màu che khuất nhau thì vẫn có thể nhầm. Để giải quyết triệt để thì cần dùng thuật toán kết hợp đặc trưng hình ảnh như DeepSORT."

### ❓ Q5: "Đếm qua vạch dùng công thức gì để xác định?"
> "Em dùng công thức **tích chéo** trong hình học vector để xét xem tọa độ tâm của sản phẩm đang nằm bên trái hay bên phải vạch kẻ. Bằng cách so sánh vị trí tâm ở khung hình trước và khung hình hiện tại, nếu vị trí này đổi dấu (tức là chuyển từ trái sang phải hoặc ngược lại) thì em xác định sản phẩm đã cắt ngang qua vạch, và tăng số đếm lên 1. Để tránh đếm lặp, mỗi đối tượng đều có cờ trạng thái 'đã đếm'."

### ❓ Q6: "Phân ngưỡng Otsu hoạt động thế nào?"
> "Thuật toán Otsu dựa trên giả định rằng biểu đồ cường độ ánh sáng của ảnh có dạng **hai đỉnh rõ rệt** (một đỉnh của nền, một đỉnh của đối tượng). Nó sẽ quét nghiệm qua tất cả các giá trị ngưỡng từ 0 đến 255, và tính **phương sai liên lớp** tại mỗi giá trị. Ngưỡng nào đem lại phương sai liên lớp lớn nhất sẽ được chọn. Về bản chất, nó tìm ra vạch chia để hai phần ảnh phân tách rõ rệt nhất."

### ❓ Q7: "Hãy so sánh phương pháp xử lý ảnh truyền thống và phương pháp học sâu YOLOv8?"

| Tiêu chí | Phương pháp truyền thống | Phương pháp học sâu (YOLOv8) |
|-----------|--------------------------|-------------------|
| **Dữ liệu huấn luyện**| Không cần huấn luyện | Cần thu thập ảnh và huấn luyện |
| **Tốc độ xử lý** | Rất nhanh (khoảng 40 khung hình/giây trên CPU) | Chậm hơn nhiều trên CPU (chỉ khoảng 10-15 khung hình/giây) |
| **Độ chính xác** | Khá nhạy cảm với ánh sáng và cần tinh chỉnh tham số | Cao hơn, ổn định hơn và chống chịu nhiễu tốt |
| **Yêu cầu phần cứng**| Nhẹ, chạy tốt trên máy yếu | Cần có card đồ họa (GPU) để đạt tốc độ thời gian thực |

### ❓ Q8: "Em xây dựng giao diện bằng thư viện gì?"
> "Em dùng thư viện **CustomTkinter**. Đây là thư viện mở rộng từ Tkinter truyền thống của Python, cung cấp các thành phần giao diện hiện đại, các góc bo tròn, và đặc biệt là hỗ trợ giao diện nền tối (dark mode) rất đẹp mắt. Do nó dựa trên nền tảng Tkinter có sẵn, việc cài đặt và triển khai rất nhẹ nhàng."

---

## PHẦN 4: CÁC THÔNG SỐ QUAN TRỌNG CẦN NHỚ TRONG HỆ THỐNG

| Tên tham số | Giá trị cài đặt | Ý nghĩa thực tế |
|---------|---------|---------|
| Số khung hình học nền | 300 | Số lượng khung hình dùng để xây dựng mô hình nền tĩnh |
| Ngưỡng phương sai phân loại | 36 | Mức ngưỡng để quyết định xem điểm ảnh thuộc nền hay thuộc tiền cảnh |
| Kích thước phần tử cấu trúc | Vuông 11×11 | Kích thước ma trận trượt để quét thực hiện phép co/nở |
| Số lần lặp phép hình thái | 3 | Số lần áp dụng liên tiếp phép co/nở |
| Diện tích tối thiểu | 600 | Số điểm ảnh tối thiểu để một vệt được công nhận là sản phẩm |
| Diện tích tối đa | 60000 | Diện tích lớn nhất của một sản phẩm đơn lẻ (dùng để phát hiện vật dính nhau) |
| Khoảng cách ghép tối đa | 60 | Khoảng cách giới hạn (tính bằng điểm ảnh) để ghép vết theo dõi |
| Số khung hình chờ mất tích | 15 | Số khung hình tối đa chờ đợi trước khi quyết định xóa sổ một đối tượng |
| Lọc mờ Gauss | 5×5, độ lệch chuẩn 1.2 | Lọc nhiễu trước khi phân ngưỡng |

---

## PHẦN 5: BẢNG TRA CỨU NHANH THUẬT NGỮ CHUYÊN NGÀNH

| Tiếng Việt đã dùng trong báo cáo | Thuật ngữ gốc (Tiếng Anh) |
|-----------|-----------|
| Đường ống xử lý / Luồng xử lý | Pipeline |
| Khung hình | Frame |
| Điểm ảnh | Pixel |
| Vùng quan tâm | ROI (Region of Interest) |
| Phân vùng ảnh | Segmentation |
| Trừ nền | Background Subtraction |
| Tiền cảnh (đối tượng di chuyển) | Foreground |
| Nền tĩnh | Background |
| Phân ngưỡng | Threshold / Thresholding |
| Ảnh nhị phân / Mặt nạ | Binary Image / Mask |
| Ảnh xám / Ảnh đa mức xám | Grayscale |
| Phép toán hình thái học | Morphological Operations |
| Phép co / Phép xâm thực | Erosion |
| Phép nở / Phép giãn nở | Dilation |
| Phép mở | Opening |
| Phép đóng | Closing |
| Phần tử cấu trúc / Ma trận nhân | Kernel / Structuring Element |
| Đường viền | Contour |
| Phát hiện đường viền | Contour Detection |
| Hình chữ nhật bao quanh | Bounding Box |
| Tọa độ tâm / Trọng tâm | Centroid |
| Diện tích | Area |
| Theo dõi đối tượng | Tracking |
| Vết theo dõi | Track |
| Cắt qua vạch / Vượt qua vạch đếm | Line Crossing |
| Tích chéo | Cross Product |
| Sắc độ (loại màu) | Hue |
| Độ bão hòa (đậm/nhạt) | Saturation |
| Độ sáng | Value |
| Phân loại màu sắc | Color Classification |
| Mô hình hỗn hợp Gauss | Gaussian Mixture Model (GMM) |
| Ma trận khoảng cách | Distance Matrix |
| Thuật toán gán tham lam | Greedy Assignment |
| Khoảng cách hình học thẳng | Euclidean Distance |
| Học sâu | Deep Learning |
| Chạy suy luận (chạy mô hình) | Inference |
| Huấn luyện mô hình | Training |
| Mô hình được huấn luyện sẵn | Pretrained Model |
| Biến đổi phân thủy | Watershed Transform |
| Biến đổi khoảng cách | Distance Transform |
| Biểu đồ phân bố cường độ | Histogram |
| Dạng hai đỉnh | Bimodal |
| Phương sai liên lớp | Between-class Variance |
| Giao diện đồ họa người dùng | GUI (Graphical User Interface) |
| Giao diện nền tối | Dark Mode |
| Số khung hình mỗi giây | FPS (Frames Per Second) |
