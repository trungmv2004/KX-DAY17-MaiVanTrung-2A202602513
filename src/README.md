# Memory Systems for AI Agent

Phần triển khai của Lab 17 chạy offline, tất định và không cần API key.

| File | Trách nhiệm |
| --- | --- |
| `config.py` | Nạp `.env`, đường dẫn repo và cấu hình model/judge; mặc định compact ở 1.200 token, giữ 4 message |
| `model_provider.py` | Chuẩn hóa alias và khởi tạo model cho sáu provider bằng import theo nhu cầu |
| `memory_store.py` | `User.md`, metadata `Memory.json`, confidence threshold, decay theo lượt, correction và compact |
| `offline_responses.py` | Chính sách trả lời chung cho hai agent và lời gọi model ở chế độ live |
| `agent_baseline.py` | Lưu toàn bộ lịch sử theo thread; không có persistent memory hoặc compact |
| `agent_advanced.py` | Profile theo user, lịch sử theo thread và summary có giới hạn kích thước |
| `benchmark.py` | Hai dataset, recall ở thread mới, sáu chỉ số, xuất Markdown/JSON và phép thử tắt compact |
| `bonus_benchmark.py` | So sánh cập nhật correction với chính sách giữ fact đầu tiên về nơi ở/nghề nghiệp |
| `bonus_policies.py` | Hai phép thử riêng cho confidence threshold và memory decay |
| `live_smoke.py` | 10 kiểm tra với API thật; xuất JSON, giới hạn request và không cho phép fallback offline |
| `test_agents.py` | Kiểm chứng memory, cách ly user, token accounting, provider và benchmark trên dữ liệu thật |

Chạy tại thư mục gốc repo, sau khi kích hoạt `.venv`:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
python src/bonus_benchmark.py
python src/bonus_policies.py
python src/live_smoke.py --provider gemini --model gemini-3.1-flash-lite --output results/live_gemini.json
```

Benchmark dùng một thư mục sạch riêng cho mỗi dataset trong `state/benchmarks/`. `data/` chỉ được đọc. Không cần xóa toàn bộ `state/` để chạy lại.

Lệnh `live_smoke.py` cần key phù hợp và gọi API thật; các lệnh khác chạy offline. Bộ pytest không gọi API. Gemini đã đạt 10/10 kiểm tra live; các adapter còn lại chỉ được kiểm tra khởi tạo.

Advanced chỉ ghi candidate đạt `MEMORY_CONFIDENCE_THRESHOLD` (mặc định `0.8`). Confidence là điểm bằng quy tắc, chưa phải xác suất. `Memory.json` giữ confidence, `last_seen_turn`, `seen_count` và lượt hiện tại theo user, tồn tại qua restart. Priority giảm theo `confidence × 2^(-age / half_life)` với half-life mặc định 100 lượt. Fact dưới `MEMORY_DECAY_MIN_PRIORITY=0.2` bị loại khỏi phần profile trong prompt; tên không giảm theo tuổi. `User.md` được giữ lại, xác nhận mới khôi phục priority. Có thể tắt bằng `MEMORY_DECAY_ENABLED=false`.

Memory growth của Advanced tính cả `User.md` và `Memory.json`. Decay giảm số fact đưa vào prompt, không giảm dung lượng file. Lịch sử và summary trong thread vẫn có thể chứa thông tin cũ; phép thử decay dùng thread mới để kiểm tra riêng persistent memory.

`reply()` trả dict gồm `answer`, `agent_tokens` và `prompt_tokens`; hai chỉ số token là số tăng thêm ở lượt hiện tại. Các phương thức `token_usage()` và `prompt_token_usage()` trả số cộng dồn theo thread. Cả hai agent dùng cùng estimator và cùng system prompt.

Offline dùng regex và heuristic, không giả lập khả năng suy luận của một LLM. Live gọi model qua giao diện LangChain `.invoke()` và dùng cùng luồng quản lý memory. Model chỉ được dựng khi bật live, có model thực và có key phù hợp; thiếu key thì agent dùng nhánh offline. Lỗi API thực được trả về cho người gọi. Bản này không dùng tool agent tự ghi profile hoặc LLM summarizer; việc ghi fact và compact do Python kiểm soát.

Xem [phân tích kết quả](../ANALYSIS.md), [kết quả benchmark](../results/benchmark.md) và [cấu hình mẫu](../.env.example).
