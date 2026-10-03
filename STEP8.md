# Bước 8 — Phân tích kết quả benchmark

Câu trả lời cho bốn yêu cầu ở **Bước 8** trong [Guide.md](Guide.md). Số liệu lấy từ [benchmark.json](results/benchmark.json), chạy offline với ngưỡng compact **1.800 token**, giữ **4 message** gần nhất. Token được ước lượng bằng heuristic ký tự, không phải token tính phí của API.

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Recall | Memory growth (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 1846 | 21642 | 0.000 | 0 | 0 |
| Standard | Advanced | 2049 | 29504 | 1.000 | 1215 | 0 |
| Stress | Baseline | 337 | 23695 | 0.000 | 0 | 0 |
| Stress | Advanced | 394 | 19091 | 1.000 | 993 | 1 |

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ lịch sử theo `thread_id`. Khi hỏi recall ở thread mới, agent không còn thông tin từ hội thoại trước nên recall bằng **0** trên cả hai bộ dữ liệu.

Advanced trích các fact ổn định như tên, nơi ở, nghề nghiệp và phong cách trả lời, sau đó lưu vào `User.md` theo `user_id`. Trong thread mới, agent đọc các fact còn đủ ưu tiên để trả lời. Vì vậy Advanced đạt recall **1,0** trên cả Standard và Stress; thông tin vẫn tồn tại khi tạo lại instance agent.

Recall ở đây được chấm bằng chuỗi con mong đợi. Kết quả chỉ phản ánh hai bộ dữ liệu đã chạy, chưa chứng minh agent hiểu đúng mọi cách diễn đạt.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trên Standard, Advanced xử lý **29.504 prompt token**, cao hơn **21.642** của Baseline khoảng **36,3%**. Hội thoại chưa vượt ngưỡng compact nên cả hai đều có **0 compaction**; Advanced phải mang thêm profile vào mỗi prompt mà chưa nhận được lợi ích từ nén lịch sử.

Advanced cũng sinh **2.049 output token**, so với **1.846** của Baseline, vì trả lời được câu hỏi recall bằng thông tin cụ thể. Việc ghi file không trực tiếp tạo token; chi phí ngữ cảnh tăng khi nội dung profile được đưa vào prompt.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Baseline giữ toàn bộ lịch sử, nên thông tin cũ được gửi lại qua nhiều lượt. Advanced tóm tắt các message cũ khi vượt ngưỡng và giữ các message gần nhất, giúp giảm lượng ngữ cảnh phải xử lý.

Trên Stress, Advanced có **1 compaction**, xử lý **19.091 prompt token**, giảm **19,4%** so với **23.695** của Baseline và vẫn đạt recall **1,0**.

Phép thử [tắt compact](results/benchmark_no_compact.json) cho thấy chính Advanced cần **25.101 prompt token** khi không nén. Bật compact giảm **23,9%** so với đối chứng này; output token vẫn là **394** và recall vẫn **1,0**. Điều này cho thấy lợi ích chính của compact nằm ở **Prompt tokens processed**. Tuy nhiên, summary có thể bỏ mất chi tiết tạm thời; recall trong bộ này chủ yếu hỏi fact đã lưu trong profile.

## 4. File memory tăng trưởng ra sao và có rủi ro gì?

Baseline không ghi persistent memory nên tăng **0 byte**. Advanced tăng tổng cộng **1.215 byte** trên Standard và **993 byte** trên Stress. Phép đo tính cả `User.md` và metadata `Memory.json`:

| Bộ dữ liệu | User.md (bytes) | Memory.json (bytes) | Tổng (bytes) |
| --- | ---: | ---: | ---: |
| Standard | 347 | 868 | 1215 |
| Stress | 435 | 558 | 993 |

Fact mới thêm một khóa; correction thay giá trị của khóa hiện có để tránh lưu đồng thời thông tin cũ và mới. `Memory.json` lưu confidence, lượt xác nhận gần nhất và số lần xác nhận. Memory decay loại fact hết ưu tiên khỏi phần profile trong prompt, nhưng giữ dữ liệu gốc nên **không làm giảm dung lượng file**.

Các rủi ro chính:

- **Lưu sai hoặc ghi đè fact đúng:** regex có thể nhận diện sai. Confidence threshold mặc định `0.8` lọc một số câu không chắc chắn, trích dẫn và thông tin người khác, nhưng điểm này chưa phải xác suất đã hiệu chỉnh.
- **Quên thông tin còn đúng:** decay quá nhanh có thể giảm recall; người dùng xác nhận lại sẽ khôi phục priority.
- **Mất chi tiết khi compact:** summary có giới hạn kích thước nên có thể bỏ sót thông tin cần cho câu hỏi tiếp theo.
- **Thông tin riêng tư và tính nhất quán:** cần cơ chế sửa/xóa và kiểm soát truy cập khi triển khai thực tế. Bản hiện tại cách ly user và ghi từng file qua file tạm, nhưng chưa có transaction chung cho hai file hoặc khóa khi nhiều tiến trình cùng ghi.

Dung lượng trên là mức tăng cuối trừ đầu của persistent memory; không đo RAM, summary hoặc tổng số byte đã ghi qua mọi lượt. Phân tích đầy đủ nằm trong [ANALYSIS.md](ANALYSIS.md).
