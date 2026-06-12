Để đề cương đồ án của nhóm thực sự **nổi bật** và thuyết phục được Thầy/Cô (đặc biệt là khi nhóm nhấn mạnh việc chỉ dùng xử lý ảnh truyền thống ), bạn cần bổ sung các giải pháp để giải quyết những "gót chân Achilles" kinh điển của phương pháp này.

Dưới đây là các điểm cải tiến đáng giá, chia theo từng module mà nhóm có thể đưa vào để nâng tầm đề tài:

---

## 1. Cải tiến Pipeline Xử lý ảnh (Khắc phục điểm yếu môi trường)

Các phương pháp truyền thống rất nhạy cảm với ánh sáng và nhiễu. Nhóm có thể đề xuất thêm:

* 
**Cân bằng sáng tự động (Adaptive Thresholding/CLAHE):** Thay vì phân ngưỡng cố định, hãy đề xuất dùng phân ngưỡng thích nghi hoặc thuật toán CLAHE để hệ thống vẫn hoạt động tốt khi ánh sáng nhà xưởng thay đổi (sáng tối thất thường).


* 
**Xử lý bóng đổ (Shadow Removal):** Khi sản phẩm đi qua camera thường có bóng đổ, dễ bị thuật toán hiểu lầm là một phần của sản phẩm (làm sai lệch contour). Nhóm nên đề xuất chuyển sang hệ màu **HSV** hoặc **Lab** để tách biệt kênh độ sáng (V/L) và kênh màu sắc, giúp loại bỏ bóng đổ hiệu quả hơn so với chỉ dùng ảnh xám.



---

## 2. Nâng cấp cơ chế Đếm và Tracking (Tránh đếm sai)

Cơ chế "đếm qua vạch"  rất dễ bị lỗi nếu sản phẩm đi sát nhau hoặc bị rung lắc. Hãy biến phần này thành điểm nhấn:

* 
**Thuật toán Tracking mạnh mẽ hơn:** Thay vì chỉ theo dõi vị trí cơ bản, nhóm có thể áp dụng thuật toán **Centroid Tracking** (tính toán khoảng cách Euclidean giữa tâm các vật thể qua từng frame) hoặc **Kalman Filter** để dự đoán quỹ đạo. Điều này giúp hệ thống không bị mất dấu khi một sản phẩm vô tình bị che khuất một phần nhỏ.


* 
**Cơ chế "Vùng đếm" (ROI) thay vì một vạch đơn:** Thay vì dùng 1 dòng kẻ duy nhất, hãy dùng một **vùng đếm (Counting Zone)**. Sản phẩm đi vào vùng -> kích hoạt trạng thái "đang đếm" -> đi ra khỏi vùng mới chính thức cộng 1. Cách này loại bỏ hoàn toàn lỗi một sản phẩm bị "bật nảy" qua lại trên vạch dẫn đến việc đếm trùng nhiều lần.



---

## 3. Thêm tính năng "Xử lý sản phẩm dính liền" (Advanced Feature)

Trong phạm vi nhóm có ghi "sản phẩm tách biệt", nhưng thực tế sản phẩm trên băng chuyền rất dễ bị dính vào nhau. Nếu nhóm giải quyết được một phần bài toán này bằng thuật toán truyền thống, điểm số sẽ rất cao:

* 
**Áp dụng thuật toán Watershed (Phân thủy đoạn):** Khi hai sản phẩm chạm vào nhau, đường bao (contour) sẽ gộp thành một. Nhóm có thể đề xuất dùng phép biến đổi khoảng cách (Distance Transform) kết hợp Watershed để "cắt" các sản phẩm bị dính liền ra thành các thực thể độc lập trước khi đếm.



---

## 4. Bổ sung Tính năng Phụ trợ cho GUI (Tăng tính thực tế)

Để giao diện (GUI) không chỉ dừng lại ở mức "dễ thao tác", hãy biến nó thành một bảng điều khiển (Dashboard) công nghiệp thực sự:

* 
**Nút "Set Background" (Lấy mẫu nền):** Cho phép người vận hành bấm nút để chụp lại băng chuyền trống làm nền chuẩn trước khi chạy, giúp bước trừ nền (Background Subtraction) đạt độ chính xác tuyệt đối.


* **Tính năng cài đặt kích thước chuẩn (Calibration):** Cho phép người dùng vẽ một ô vuông trên màn hình và nhập kích thước thật (ví dụ: ô này là $10\text{cm} \times 10\text{cm}$). Từ đó, hệ thống không chỉ đếm mà còn **ước lượng được kích thước thực tế** của sản phẩm để loại bỏ phế liệu (quá nhỏ hoặc quá to).


* 
**Xuất báo cáo tự động:** Thêm tính năng xuất kết quả đếm ra file Excel (`.csv`) kèm theo mốc thời gian (Timestamp). Đây là tính năng mà bất kỳ nhà máy nào cũng cần.



---

## Gợi ý cách viết vào Đề cương để thuyết phục Thầy/Cô:

> 
> "Mặc dù giới hạn đề tài ở các phương pháp xử lý ảnh truyền thống, nhóm không chỉ áp dụng các thuật toán cơ bản mà sẽ tập trung tối ưu hóa pipeline để giải quyết các bài toán thực tế của nhà xưởng như: **tự động thích nghi với sự thay đổi ánh sáng (dùng CLAHE/HSV)**, **xử lý sản phẩm dính nhau bằng Watershed**, và **chống đếm trùng bằng cơ chế vùng đếm thông minh (ROI Tracking)**. Đồng thời, GUI của nhóm sẽ tích hợp tính năng **xuất báo cáo dữ liệu theo thời gian thực** để tiệm cận với một sản phẩm công nghiệp hoàn chỉnh."
> 
> 

Những điểm cải tiến này vừa bám sát kiến thức môn học, vừa chứng minh được tư duy giải quyết vấn đề logic và thực tế của nhóm!-
