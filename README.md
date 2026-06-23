<div align="center">
  <h1>HỆ THỐNG TỰ ĐỘNG ĐẾM SỐ LƯỢNG SẢN PHẨM TRÊN BĂNG CHUYỀN</h1>
  <p>
    Ứng dụng Thị giác máy tính (Computer Vision) kết hợp Học sâu (Deep Learning YOLOv8) để theo dõi, phân loại và đếm sản phẩm trong thời gian thực.
  </p>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
  [![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://ultralytics.com/)
  [![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-blueviolet.svg)](https://customtkinter.tomschimansky.com/)
</div>

---

## 1. GIỚI THIỆU TỔNG QUAN

Hệ Thống Đếm Sản Phẩm Băng Chuyền là sản phẩm đồ án môn học Xử lý ảnh, được xây dựng nhằm ứng dụng công nghệ xử lý ảnh số và trí tuệ nhân tạo vào môi trường sản xuất công nghiệp thực tế. Tại các nhà máy, việc kiểm đếm và phân loại sản phẩm trên các dây chuyền vận chuyển thường tốn rất nhiều thời gian và nhân lực nếu thực hiện bằng phương pháp thủ công, đồng thời dễ dẫn đến sai sót do yếu tố con người. 

Hệ thống của chúng tôi được thiết kế để kết nối trực tiếp với luồng Camera giám sát tại nhà xưởng hoặc đọc dữ liệu từ các file video có sẵn. Thông qua quá trình tiếp nhận khung hình, tiền xử lý, phân ngưỡng và trích xuất đặc trưng, phần mềm có khả năng tự động nhận diện và theo dõi quỹ đạo di chuyển của từng sản phẩm riêng biệt.

Đặc biệt, phần mềm cung cấp sự linh hoạt cao khi cho phép người vận hành chuyển đổi giữa hai chế độ thuật toán cốt lõi:
- **Chế độ Xử lý ảnh truyền thống (OpenCV):** Tập trung vào việc tối ưu hóa hiệu năng phần cứng, sử dụng kỹ thuật tách nền (Background Subtraction), lọc nhiễu hình thái học, phân loại màu sắc bằng không gian HSV, và giải quyết triệt để bài toán các vật thể dính liền nhau bằng thuật toán phân thủy (Watershed) kết hợp biến đổi khoảng cách.
- **Chế độ Học sâu (YOLOv8):** Sử dụng mô hình mạng nơ-ron tích chập (CNN) hiện đại để cung cấp độ chính xác cao trong môi trường có bối cảnh phức tạp, ánh sáng thay đổi liên tục hoặc hàng hóa có hình dạng bất quy tắc.

## 2. TÍNH NĂNG NỔI BẬT

- **Giao diện người dùng hiện đại:** Ứng dụng được xây dựng hoàn toàn bằng thư viện `CustomTkinter` với bảng điều khiển trực quan, thiết kế phẳng, thân thiện với người dùng và hỗ trợ chế độ nền tối (Dark Mode) để giảm thiểu tình trạng mỏi mắt cho người vận hành trong nhà máy.
- **Tương tác và thiết lập linh hoạt:** Cho phép kỹ thuật viên dễ dàng sử dụng chuột để vẽ Vùng quan tâm (ROI - Region of Interest) nhằm thu hẹp khu vực xử lý, loại bỏ nhiễu ngoại cảnh; đồng thời kẻ Vạch đếm (Counting Line) ở bất kỳ góc độ nào trên khung hình.
- **Phân loại màu sắc thông minh:** Không chỉ dừng lại ở việc đếm tổng số lượng, hệ thống còn tự động trích xuất đặc trưng màu sắc dựa trên dải ngưỡng HSV để phân loại và đếm tách biệt sản phẩm theo từng nhóm (Ví dụ: nhóm Đỏ, Vàng, Xanh lá...).
- **Kiến trúc Xử lý đa luồng (Multi-threading):** Nhằm giải quyết tình trạng đơ, treo giao diện (UI freezing) khi chạy các mô hình tính toán nặng như YOLOv8, ứng dụng đã được thiết kế chạy trên nhiều luồng độc lập, đảm bảo luồng giao diện luôn mượt mà.
- **Dashboard Thống kê Real-time:** Số lượng sản phẩm được tính toán và đẩy trực tiếp lên bảng Dashboard theo thời gian thực (Real-time), không có độ trễ, giúp cấp quản lý dễ dàng nắm bắt sản lượng tức thời.
- **Cấu hình tham số chuyên sâu:** Bảng Settings chuyên dụng cho phép các kỹ thuật viên tùy chỉnh toàn bộ thông số thuật toán (Dải màu HSV, Ngưỡng diện tích nhỏ nhất/lớn nhất, Số khung hình học nền tĩnh, Chu kỳ xóa ID đối tượng mất dấu) ngay trong lúc hệ thống đang chạy mà không cần khởi động lại.

## 3. CÔNG NGHỆ VÀ THƯ VIỆN SỬ DỤNG

- **Ngôn ngữ lập trình:** Python (Phiên bản 3.8 trở lên).
- **Thị giác máy tính và Xử lý ảnh:** Thư viện OpenCV (`cv2`) và NumPy.
- **Trí tuệ nhân tạo và Học sâu:** Nền tảng Ultralytics YOLOv8.
- **Giao diện đồ họa (GUI):** Thư viện CustomTkinter và Pillow.
- **Quản lý đa luồng:** Thư viện tiêu chuẩn `threading` của Python.

## 4. HƯỚNG DẪN CÀI ĐẶT

Hệ thống tương thích tốt với các hệ điều hành Windows, macOS và Linux. Để triển khai ứng dụng trên máy tính cục bộ, vui lòng làm theo các bước sau:

**Bước 1: Clone kho lưu trữ mã nguồn**
Mở Terminal hoặc Command Prompt và chạy lệnh:
```bash
git clone https://github.com/your-username/Conveyyor_Belt_Counting.git
cd Conveyyor_Belt_Counting
```

**Bước 2: Thiết lập môi trường ảo (Virtual Environment)**
Việc sử dụng môi trường ảo được khuyến nghị mạnh mẽ để tránh xung đột thư viện với các dự án Python khác trên máy tính của bạn.
```bash
python -m venv venv
# Đối với hệ điều hành Windows:
venv\Scripts\activate
# Đối với hệ điều hành macOS/Linux:
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện phụ thuộc**
Hệ thống sử dụng các thư viện đã được liệt kê trong file `requirements.txt`. Tiến hành cài đặt tự động bằng lệnh:
```bash
pip install -r requirements.txt
```

*(Lưu ý: Nếu bạn có ý định chạy ứng dụng bằng thuật toán YOLOv8, vui lòng đảm bảo file trọng số mô hình đã huấn luyện `best_yolo_model.pt` được đặt đúng vào bên trong thư mục `assets/`).*

## 5. HƯỚNG DẪN SỬ DỤNG

Sau khi quá trình cài đặt hoàn tất, bạn có thể khởi động giao diện chính của phần mềm bằng lệnh sau:
```bash
python run_app.py
```

**Quy trình vận hành cơ bản dành cho người dùng:**
1. **Khởi tạo dữ liệu:** Tại giao diện chính, nhấn nút "Chọn nguồn đầu vào" để mở hộp thoại và tải lên một file video (.mp4, .avi) có sẵn trên máy tính, hoặc chọn kết nối với Webcam. Khung hình đầu tiên của video sẽ được hiển thị.
2. **Khoanh vùng xử lý:** Nhấn nút "Vẽ ROI", sau đó sử dụng chuột để kéo thả một hình chữ nhật bao quanh khu vực băng chuyền. Hệ thống sẽ chỉ quét và xử lý các điểm ảnh nằm trong vùng này.
3. **Thiết lập ranh giới:** Nhấn nút "Kẻ Vạch đếm", click chuột để chọn điểm đầu và điểm cuối nhằm vạch ra một đường thẳng ngang băng chuyền.
4. **Lựa chọn giải thuật:** Tùy thuộc vào cấu hình máy tính và môi trường ánh sáng, lựa chọn thuật toán OpenCV hoặc YOLOv8 tại menu xổ xuống trên thanh công cụ.
5. **Tiến hành tự động hóa:** Nhấn nút "Start" để hệ thống bắt đầu quá trình đọc khung hình, theo dõi và đếm. Trong quá trình chạy, người vận hành có thể nhấn "Pause" để tạm dừng và phân tích, hoặc "Stop" để kết thúc phiên làm việc.
6. **Theo dõi kết quả:** Mọi thông tin về ID sản phẩm, Bounding Box bao quanh, nhãn phân loại màu sắc sẽ được hiển thị trực tiếp lên video. Đồng thời số liệu sản lượng sẽ nhảy liên tục trên bảng Dashboard bên phải.

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Conveyor_Belt_Counting/
├── run_app.py                 # File thực thi chính để khởi động ứng dụng
├── requirements.txt           # Danh sách các gói thư viện Python cần thiết
├── README.md                  # File tài liệu hướng dẫn tổng quan (đang đọc)
├── conveyor_counter/          # Thư mục mã nguồn lõi (Source code)
│   ├── app.py                 # File quản lý toàn bộ giao diện và các sự kiện UI
│   ├── vision.py              # Xử lý các phép toán OpenCV (Lọc nhiễu, Tách nền, Watershed)
│   ├── tracker.py             # Thuật toán tính toán khoảng cách tâm và liên kết ID
│   ├── yolo_detector.py       # Lớp quản lý Worker Thread cho mô hình suy luận YOLOv8
│   ├── config.py              # Đọc/Ghi file cấu hình tham số hệ thống dạng JSON
│   └── geometry.py            # Chứa các hàm toán học vector để kiểm tra trạng thái qua vạch
├── assets/                    # Thư mục chứa các tài nguyên (Video demo, Model Weights .pt)
├── tests/                     # Thư mục chứa các script kiểm thử tự động (Unit Test)
└── diagrams/                  # Thư mục chứa tài liệu thiết kế hệ thống và sơ đồ UML (.puml)
```

## 7. THÔNG TIN NHÓM THỰC HIỆN

Dự án này là Tiểu luận chuyên ngành thuộc môn học Xử lý Ảnh số, được thực hiện bởi Nhóm 08 (Khoa Đào tạo Chất lượng cao - Trường Đại học Công nghệ Kỹ thuật TP. Hồ Chí Minh).

- **Giảng viên hướng dẫn:** PGS.TS Hoàng Văn Dũng
- **Sinh viên thực hiện:**
  - Nguyễn Minh Quốc Khánh - MSSV: 23110113
  - Nguyễn Bách Tùng - MSSV: 23110166
  - Nguyễn Hưng Nguyên - MSSV: 23110135

Chúng em xin chân thành cảm ơn PGS.TS Hoàng Văn Dũng đã tận tình hướng dẫn và cung cấp những nền tảng kiến thức chuyên sâu để nhóm có thể hoàn thành tốt dự án này.

---

<details>
<summary><b>TÀI LIỆU ÔN TẬP VÀ TRẢ LỜI CÂU HỎI BẢO VỆ ĐỒ ÁN (Bấm vào đây để xem chi tiết)</b></summary>
  
**Câu 1: Thuật toán trừ nền MOG2 có nguyên lý hoạt động ra sao?**
Thuật toán này sử dụng một tập hợp nhiều phân phối Gauss để học và ghi nhớ các biến thiên của bối cảnh nền tĩnh theo thời gian. Mỗi khi có một điểm ảnh mới, mô hình đối chiếu nó với tập hợp phân phối đã học. Nếu nằm ngoài các phân phối, điểm đó được đánh dấu là một phần của vật thể đang di chuyển. Do MOG2 có khả năng liên tục tự học và cập nhật mô hình nền, nó có thể tự động thích nghi nếu cường độ chiếu sáng trong phòng thay đổi một cách chậm rãi.

**Câu 2: Tại sao dự án quyết định phân loại màu sắc trong không gian HSV thay vì dùng ảnh RGB?**
Bởi vì hệ màu RGB (Red, Green, Blue) gộp chung thông tin về màu sắc và ánh sáng vào cả 3 kênh. Trong môi trường nhà máy thực tế, khi độ sáng rọi vào băng chuyền thay đổi hoặc có bóng râm xuất hiện, cả R, G và B đều thay đổi theo, làm chệch quy luật phân loại màu cứng. Ngược lại, việc sử dụng mô hình HSV bóc tách ánh sáng vào kênh V (Value), qua đó ta chỉ cần giới hạn khoảng tham chiếu trên một mình kênh H (Hue - sắc độ màu) là đã đủ để lọc được màu chuẩn xác trong hầu hết mọi điều kiện ánh sáng.

**Câu 3: Vai trò của phép hình thái học (phép Mở và phép Đóng) trong ứng dụng thực tế này là gì?**
Khi thực hiện thuật toán trừ nền, kết quả ảnh nhị phân thường không hoàn hảo: hay xuất hiện các hạt lốm đốm nhiễu bên ngoài băng chuyền và bị khuyết lỗ bên trong hình dạng sản phẩm. Phép mở (tức là dùng phép co trước rồi nở sau) sẽ bào mòn hết đốm nhiễu ngoại cảnh. Trong khi đó phép đóng (nở trước rồi co lại) sẽ kéo dãn để lấp kín các khe hở, lỗ hổng bên trong. Cả hai phép toán này kết hợp giúp tạo ra một vùng đối tượng (blob) nguyên vẹn, đảm bảo tính chuẩn xác cho bước tìm đường viền.

**Câu 4: Quá trình phân tách hai sản phẩm cùng màu nằm dính sát nhau được thực hiện như thế nào?**
Một trong những lỗi kinh điển là hai sản phẩm cùng màu chạy dính liền với nhau tạo thành một khối contour lớn, dẫn đến đếm thiếu. Hệ thống tự động khắc phục bằng cách:
- Thực hiện phép biến đổi khoảng cách (Distance Transform) để tìm cực đại (chính là tâm thực sự nằm sâu bên trong từng vùng vật thể).
- Áp dụng kỹ thuật phân thủy (Watershed Transform) tràn từ các điểm tâm này ra xung quanh giống như nước chảy vào thung lũng, từ đó xác định chính xác đường ranh giới và chia đôi khối bị dính liền thành hai đối tượng độc lập.

**Câu 5: Nhóm kiểm tra điều kiện sản phẩm đi qua vạch đếm bằng nguyên lý toán học nào?**
Hệ thống thiết lập phương trình cho vạch đếm dưới dạng một đoạn thẳng nối giữa điểm đầu và điểm cuối do người dùng vẽ. Bằng việc sử dụng công thức tích chéo (cross product) giữa véc-tơ vạch đếm và tọa độ điểm tâm của vật thể, hệ thống sẽ sinh ra một giá trị vô hướng biểu diễn xem tâm vật thể đang đứng bên trái hay bên phải của vạch. Ngay tại khoảnh khắc vật thể di chuyển tiến tới làm cho dấu của giá trị này đảo ngược so với khung hình trước đó, hệ thống chốt số lượng đếm tăng lên 1 và gán cờ vô hiệu hóa để không bao giờ đếm lại đối tượng đó nữa.

**Câu 6: Tại sao cần phải dùng kỹ thuật xử lý đa luồng (Multi-threading) khi tích hợp YOLOv8?**
YOLOv8 là mô hình mạng nơ-ron học sâu yêu cầu tài nguyên xử lý rất lớn. Nếu chạy chung tác vụ suy luận (inference) này với luồng chính (Main Thread) quản lý giao diện, phần mềm sẽ bị treo hoặc đơ cứng mỗi khi phân tích xong một khung hình. Việc đưa quy trình nhận diện YOLO vào một Worker Thread riêng biệt chạy nền giúp giải phóng luồng chính, nhờ vậy giao diện (Tkinter) vẫn hiển thị mượt mà và nhận các tương tác click chuột của người vận hành một cách tức thời.

</details>
