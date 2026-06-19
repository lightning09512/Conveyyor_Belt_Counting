# Hệ Thống Đếm Sản Phẩm Băng Chuyền

Dự án này là một phần mềm hỗ trợ đếm số lượng và phân loại màu sắc sản phẩm chạy trên băng chuyền bằng kỹ thuật thị giác máy tính. Hệ thống sử dụng các phương pháp xử lý ảnh truyền thống qua OpenCV, đồng thời cung cấp tùy chọn mở rộng bằng mô hình học sâu YOLOv8 để nâng cao độ chính xác.

## Cấu trúc thư mục

```text
Conveyyor_Belt_Counting/
├── run_app.py                 # File khởi động ứng dụng
├── requirements.txt
├── conveyor_config.example.json
├── TRAINING_GUIDE.md
├── conveyor_counter/          # Thư mục mã nguồn chính
│   ├── app.py                 # Giao diện đồ họa (GUI)
│   ├── vision.py              # Luồng xử lý ảnh truyền thống
│   ├── tracker.py             # Thuật toán theo dõi và đếm
│   ├── yolo_detector.py       # Tích hợp mô hình YOLOv8
│   ├── config.py
│   └── geometry.py
├── tools/                     # Các script hỗ trợ
├── tests/
└── assets/                    # Thư mục chứa video, model, dataset (chạy local)
    ├── demo.mp4
    ├── demo_config.json
    ├── best_yolo_model.pt
    └── yolo_dataset/
```

## Hướng dẫn cài đặt và chạy ứng dụng

```powershell
pip install -r requirements.txt
python run_app.py
```

- Tệp cấu hình mẫu: `conveyor_config.example.json` hoặc `assets/demo_config.json`
- Video thử nghiệm: `assets/demo.mp4`
- Để huấn luyện mô hình YOLOv8, vui lòng xem hướng dẫn trong `TRAINING_GUIDE.md`.
- Để chạy các bài test: `python -m unittest discover -s tests`

---

# TÀI LIỆU ÔN TẬP BÁO CÁO ĐỒ ÁN

Tài liệu dưới đây tổng hợp lại toàn bộ luồng hoạt động của hệ thống và các câu hỏi thường gặp để hỗ trợ cho việc bảo vệ đồ án trước giảng viên một cách tự nhiên và rõ ràng nhất.

## Phần 1: Đường Ống Xử Lý (Pipeline)

Hệ thống hoạt động theo một luồng 7 bước cơ bản. Đây là nội dung cốt lõi để trả lời cho câu hỏi "Hệ thống hoạt động như thế nào từ lúc nhận ảnh đến lúc đếm xong?".

Luồng xử lý từ đầu đến cuối:
Khung hình -> Cắt vùng quan tâm -> Phân vùng ảnh -> Xử lý hình thái học -> Phát hiện đường viền -> Phân loại màu sắc -> Theo dõi đối tượng -> Đếm qua vạch.

### Bước 1: Cắt vùng quan tâm (ROI)
Mục đích: Chỉ giữ lại khu vực băng chuyền cần phân tích, loại bỏ bối cảnh thừa bên ngoài để giảm nhiễu và tối ưu thời gian tính toán.
- Người dùng vẽ trực tiếp vùng quan tâm trên giao diện đồ họa.
- Toàn bộ các thuật toán phát hiện và nhận dạng sau đó chỉ áp dụng trên vùng được chọn này.

### Bước 2: Phân vùng ảnh (Tách nền)
Mục đích: Tách tiền cảnh (sản phẩm) ra khỏi nền (băng chuyền tĩnh), kết quả tạo ra là một ảnh nhị phân (mask) với vật thể màu trắng và nền màu đen.

Hai phương pháp được áp dụng:
- Trừ nền (MOG2): Phù hợp cho video có nền chuyển động nhẹ. Thuật toán này dùng hỗn hợp phân phối Gauss để học nền tĩnh theo thời gian. Các tham số chính bao gồm history (300) và varThreshold (36).
- Phân ngưỡng Otsu: Dùng khi băng chuyền có nền tĩnh hoàn toàn hoặc dùng hình ảnh. Phương pháp sẽ tự động tìm giá trị ngưỡng tốt nhất để chia tách 2 nhóm điểm ảnh bằng cách tối đa hóa phương sai liên lớp.

### Bước 3: Phép toán hình thái học
Mục đích: Khử nhiễu cho ảnh nhị phân thu được ở bước 2.
- Phép mở (Co rồi Nở): Giúp làm rụng đi các đốm nhiễu nhỏ li ti nằm bên ngoài vật thể.
- Phép đóng (Nở rồi Co): Lấp kín các lỗ hổng bên trong đối tượng.
- Cấu hình mặc định sử dụng ma trận nhân vuông kích thước 11x11 và lặp 3 lần.

### Bước 4: Phát hiện đường viền (Contour)
Mục đích: Trích xuất vị trí, ranh giới và diện tích của từng sản phẩm.
- Sử dụng hàm findContours với cờ RETR_EXTERNAL để chỉ lấy các đường viền bao quanh ngoài cùng.
- Loại bỏ các nhiễu lớn bằng cách giới hạn diện tích vật thể trong khoảng từ 600 đến 60000 pixel.
- Ở bước này, tọa độ tâm và hình chữ nhật bao quanh vật thể được tính toán để phục vụ bước theo dõi.

### Bước 5: Phân loại màu sắc
Mục đích: Gán nhãn các màu (Đỏ, Vàng, Xanh lá, Xanh dương) cho sản phẩm dựa trên không gian màu HSV.
- Lý do chọn HSV thay cho RGB: Không gian màu RGB trộn lẫn cả độ sáng và sắc độ. Trong môi trường thực tế, khi độ sáng rọi vào băng chuyền thay đổi, các giá trị RGB biến thiên rất khó dự đoán. Ngược lại, HSV cô lập phần màu sắc vào kênh H (Sắc độ), giúp mô hình nhận diện ổn định với biến đổi ánh sáng.
- Ứng dụng: Xác nhận kênh màu bằng khoảng giá trị quy ước kết hợp với điều kiện độ bão hòa (S) và độ sáng (V) đạt trên 50.

### Bước 6: Theo dõi đối tượng (Tracking)
Mục đích: Ghi nhớ quỹ đạo của sản phẩm khi nó di chuyển qua các khung hình liên tiếp để tránh đếm trùng.
- Dùng thuật toán tham lam ghép cặp dựa trên việc tìm khoảng cách hình học ngắn nhất giữa tập hợp tâm cũ và tâm mới.
- Xử lý tình trạng vật che khuất: Nếu hai tâm tìm được khác màu nhau, hệ thống cộng thêm khoảng cách phạt rất lớn, giúp phòng ngừa việc tráo đổi ID khi các sản phẩm chạy ngang qua nhau.
- Quản lý vòng đời: Nếu một vật thể khuất khỏi khung hình vượt quá 15 frame, hệ thống tự động xóa vết theo dõi đó.

### Bước 7: Đếm qua vạch
Mục đích: Tăng bộ đếm khi tâm vật thể đi cắt ngang qua vạch kiểm tra.
- Ứng dụng hình học vector (cụ thể là tích chéo) để tính giá trị xác định xem điểm tâm nằm bên trái hay bên phải vạch.
- Nếu giá trị này đổi dấu giữa khung hình trước và khung hình hiện tại, đối tượng được xác định là đã vượt qua vạch.
- Biến trạng thái cờ sẽ đánh dấu "đã đếm" để sản phẩm không bao giờ bị đếm lần thứ hai.

---

## Phần 2: Các Kỹ Thuật Chuyên Sâu Của Hệ Thống

### 1. Phân vùng kết hợp theo màu sắc
Thay vì chỉ dùng ảnh nhị phân tổng quát, hệ thống xây dựng các mặt nạ nhị phân dựa theo từng dải màu. Nhờ kết hợp (phép AND logic) giữa các mặt nạ màu riêng biệt và mặt nạ nền di chuyển, hệ thống dễ dàng phân tách thành công hai vật dính sát nhau nếu như chúng có màu sắc khác biệt hoàn toàn.

### 2. Tách vật dính cùng màu bằng phân thủy (Watershed)
Một trong những lỗi kinh điển là hai sản phẩm cùng màu chạy dính liền với nhau, tạo thành một khối nhiễu lớn. Khi gặp diện tích khối vượt mức bình thường, hệ thống tự động:
- Thực hiện phép biến đổi khoảng cách (Distance Transform) để đánh dấu các cực đại (tâm thực sự bên trong từng vùng vật thể).
- Áp dụng kỹ thuật phân thủy (Watershed Transform) tràn từ các tâm này ra xung quanh, từ đó xác định chính xác đường ranh giới và chia đôi khối dính liền.

### 3. Nâng cấp xử lý bằng học sâu YOLOv8
Hệ thống cung cấp sẵn luồng xử lý học sâu YOLOv8 giúp nhận diện đối tượng không phụ thuộc vào điều kiện ánh sáng. Phương pháp này đưa ra bounding box và phân loại trong một luồng duy nhất, đạt độ ổn định và chính xác cao hơn hẳn. Hệ thống đã được lập trình đa luồng (multithreading) và giới hạn vùng inference cho YOLO theo kích thước ROI nhằm tối ưu trải nghiệm thời gian thực mà không gây đơ giao diện.

---

## Phần 3: Danh Sách Câu Hỏi Trả Lời Gợi Ý

**Câu 1: Thuật toán trừ nền MOG2 có nguyên lý ra sao?**
Trả lời: Thuật toán này sử dụng một tập hợp nhiều phân phối Gauss để học và ghi nhớ các biến thiên của bối cảnh nền tĩnh theo thời gian. Mỗi khi có một điểm ảnh mới, mô hình đối chiếu nó với tập hợp phân phối đã học. Nếu nằm ngoài các phân phối, điểm đó là một phần của vật thể đang di chuyển. Do liên tục tự học và cập nhật mô hình, nó tự động thích nghi nếu cường độ sáng trong phòng thay đổi chậm.

**Câu 2: Tại sao dự án không phân loại màu dựa trên ảnh RGB?**
Trả lời: Bởi vì hệ màu RGB gộp chung thông tin màu sắc và ánh sáng vào cả 3 kênh. Nếu đưa vào bóng râm, cả R, G và B đều thay đổi, làm chệch quy luật ngưỡng màu. Sử dụng mô hình HSV bóc tách ánh sáng thành kênh V, qua đó ta chỉ cần giới hạn khoảng tham chiếu trên kênh H (sắc độ màu) là đủ để lọc được màu chuẩn xác trong mọi điều kiện ánh sáng môi trường.

**Câu 3: Vai trò của phép mở và phép đóng trong ứng dụng thực tế này là gì?**
Trả lời: Khi thực hiện phép trừ nền, kết quả hay bị lốm đốm nhiễu bên ngoài băng chuyền và bị khuyết lỗ bên trong hình dạng sản phẩm. Phép mở (tức là dùng phép co trước rồi nở sau) sẽ bào mòn hết đốm nhiễu. Trong khi đó phép đóng (nở trước rồi co lại) sẽ kéo dãn để bít kín các lỗ hổng bên trong. Cả hai giúp tạo ra một vùng blob nguyên vẹn, đảm bảo tính chuẩn xác cho bước tìm đường viền.

**Câu 4: Em kiểm tra điều kiện sản phẩm đi qua vạch đếm như thế nào?**
Trả lời: Em thiết lập phương trình cho vạch đếm dưới dạng một đoạn thẳng nối giữa điểm 1 và điểm 2. Việc sử dụng công thức tích chéo giữa véc-tơ vạch đếm và tọa độ điểm tâm của vật thể sẽ sinh ra một giá trị vô hướng biểu diễn xem nó đang đứng bên trái hay phải. Ngay tại khoảnh khắc mà vật thể di chuyển khiến dấu của giá trị này thay đổi so với thời điểm trước đó, hệ thống chốt số lượng đếm tăng lên 1 và gán cờ vô hiệu hóa đếm lần hai cho đối tượng đó.

**Câu 5: Tại sao cần dùng song song cả phương pháp OpenCV truyền thống và YOLOv8?**
Trả lời: OpenCV truyền thống có điểm mạnh là không cần chuẩn bị dữ liệu học sâu, xử lý rất tốc độ trên cả máy yếu (chạy 30-40 FPS trên CPU). Tuy nhiên, nó bị ảnh hưởng bởi bóng đổ và ánh sáng gắt. YOLOv8 bổ sung năng lực chống chịu nhiễu tốt nhất, dễ mở rộng nhận diện các loại hàng hóa bất quy tắc. Việc có cả 2 giúp đem lại tính linh hoạt cho phần mềm.

**Câu 6: Em sử dụng công cụ nào để thiết kế phần mềm?**
Trả lời: Em sử dụng thư viện CustomTkinter. Mặc dù xây dựng trên cái gốc của Tkinter quen thuộc, thư viện này khắc phục hoàn toàn ngoại hình thô cứng để tạo ra giao diện bo tròn, Dark mode sắc sảo như các ứng dụng thế hệ mới, rất thích hợp để làm một dashboard quản lý cho công nghiệp.

---

## Phần 4: Cấu Hình Các Tham Số Hệ Thống

- Số khung hình học nền tĩnh: 300
- Ngưỡng phương sai tách nền MOG2: 36
- Kích thước ma trận cửa sổ hình thái học: 11x11
- Số chu kỳ lặp hình thái học: 3
- Ngưỡng diện tích loại nhiễu: 600 - 60000 pixel
- Khoảng cách nối vết theo dõi tối đa: 60 pixel
- Chu kỳ (frame) chờ xóa vết theo dõi: 15
