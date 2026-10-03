# Phase 2, Track 3, Day 17: Memory Systems for AI Agent

Phần triển khai trong `src/` đã hoàn thiện. Chạy benchmark và test theo phần **Setup môi trường** bên dưới. Kết quả đã đo nằm trong [results/benchmark.md](results/benchmark.md); phần giải thích số liệu và bonus nằm trong [ANALYSIS.md](ANALYSIS.md).

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

```
.
├── README.md        # giới thiệu track (file này)
├── Guide.md         # hướng dẫn từng bước
├── Rubric.md        # tiêu chí chấm điểm
├── data/            # dữ liệu benchmark dùng chung
│   ├── conversations.json
│   └── advanced_long_context.json
└── src/             # cấu hình, memory layer, hai agent, benchmark và test
    ├── model_provider.py
    ├── config.py
    ├── memory_store.py
    ├── agent_baseline.py
    ├── agent_advanced.py
    ├── benchmark.py
    └── test_agents.py
```

Khi chạy, agent ghi `state/profiles/<user>/User.md` và `Memory.json` vào thư mục `state/`. `Memory.json` lưu confidence và lượt xác nhận gần nhất để tính decay qua nhiều phiên. Thư mục này đã nằm trong `.gitignore`.

### Vai trò từng file trong `src/`

Các file được liệt kê theo thứ tự nên triển khai:

| File | Vai trò | Thành phần chính |
|---|---|---|
| `model_provider.py` | Khởi tạo chat model cho từng provider | `ProviderConfig`, `normalize_provider()`, `build_chat_model()` |
| `config.py` | Cấu hình chung của lab | `LabConfig` (đường dẫn, ngưỡng compact, model chính + judge), `load_config()` |
| `memory_store.py` | Lõi memory layer | `estimate_tokens()`, `UserProfileStore` (read/write/edit `User.md`), `extract_profile_updates()`, `summarize_messages()`, `CompactMemoryManager` |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread | `BaselineAgent.reply()`, `token_usage()`, `prompt_token_usage()` |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact | `AdvancedAgent.reply()`, `_reply_offline()`, `_estimate_prompt_context_tokens()`, `_offline_response()` |
| `benchmark.py` | So sánh hai agent trên hai bộ dữ liệu | `run_agent_benchmark()`, `recall_points()`, `heuristic_quality()`, `format_rows()` |
| `test_agents.py` | Kiểm chứng hành vi memory | test `User.md`, compact trigger, cross-session recall, giảm prompt load |

### Luồng xử lý một lượt của Advanced Agent

```
message người dùng
  → extract_profile_candidates()   # trích fact kèm confidence theo quy tắc
  → lọc confidence → ghi User.md + Memory.json
  → CompactMemoryManager.append()  # short-term memory, tự compact khi vượt ngưỡng
  → prompt = các fact còn đủ ưu tiên + summary + recent messages
  → sinh câu trả lời → cập nhật bộ đếm token
```

Baseline Agent chỉ giữ danh sách message theo `thread_id`. Sang thread mới, nó **phải quên** toàn bộ fact cũ.

Cả hai agent nên có **chế độ offline** cho ra kết quả lặp lại được, để benchmark và test chạy được mà không cần API key. Chế độ live (LangChain/LangGraph) là phần mở rộng.

## Dữ liệu benchmark

| File | Nội dung | Mục tiêu |
|---|---|---|
| `data/conversations.json` | 10 hội thoại khoảng 10 lượt, user `dungct`, kèm `recall_questions` | Standard benchmark: đo recall qua nhiều phiên bình thường |
| `data/advanced_long_context.json` | 1 hội thoại 16 lượt rất dài, user `dungct_stress` | Long-context stress benchmark: ép compact xảy ra nhiều lần |

Mỗi hội thoại có dạng:

```json
{
  "id": "conv-01",
  "user_id": "dungct",
  "turns": ["...", "..."],
  "recall_questions": [
    { "question": "...", "expected_contains": ["DũngCT", "cà phê sữa đá"] }
  ]
}
```

`recall_questions` được hỏi ở **thread mới**. Điểm recall dựa trên số chuỗi trong `expected_contains` xuất hiện trong câu trả lời.

Dữ liệu cố tình chứa các tình huống khó:

- **correction**: nơi ở đổi giữa Đà Nẵng và Huế, agent phải giữ fact mới nhất
- **nhiễu**: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là câu đùa
- **ngữ cảnh dài**: nhiều đoạn tin tức dài trong stress test để làm lộ chi phí prompt của baseline

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Trên Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` cố định phiên bản của các thư viện trực tiếp đã kiểm tra. Có thể chép `.env.example` thành `.env` để cấu hình. Mặc định là **offline**, không cần API key; chế độ live cần `LLM_MODE=live`, `LLM_MODEL` thực và key của provider tương ứng (Ollama không cần key). Biến môi trường đã export có ưu tiên cao hơn `.env`.

Nếu muốn chạy chế độ live với LLM thật, hãy tạo file `.env` ở root repo (đã nằm trong `.gitignore`). Tên biến môi trường do các bạn quyết định khi viết `load_config()`. Ví dụ:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

## Chạy benchmark và test

Sau khi hoàn thiện `src/`, chạy từ root repo:

```bash
python src/benchmark.py
```

```bash
pytest src/test_agents.py -v
```

`pytest.ini` đặt thư mục tạm tại `state/pytest` để các test dùng `tmp_path` chạy được trong workspace trên Windows. Thư mục này dành riêng cho test và được pytest làm sạch khi chạy.

Lưu kết quả và chạy các phép kiểm chứng bổ sung:

```bash
python src/benchmark.py --output results/benchmark.md --json results/benchmark.json
python src/benchmark.py --no-compact --output results/benchmark_no_compact.md --json results/benchmark_no_compact.json
python src/bonus_benchmark.py
python src/bonus_policies.py
```

Confidence threshold mặc định `0.8`; khai báo rõ ràng có điểm `0.98`, thông tin về người khác, trích dẫn và câu không chắc chắn có điểm thấp hơn. Đây là điểm theo quy tắc, chưa phải xác suất đã hiệu chỉnh.

Memory decay tính theo lượt của từng user: `priority = confidence × 2^(-age / half_life)`. Mặc định half-life 100 lượt, ngưỡng ưu tiên `0.2`, tên được giữ ổn định. Fact dưới ngưỡng được bỏ khỏi phần profile trong prompt; dữ liệu gốc vẫn nằm trong `User.md` và được kích hoạt lại khi người dùng xác nhận. Cấu hình qua `MEMORY_CONFIDENCE_THRESHOLD`, `MEMORY_DECAY_HALF_LIFE_TURNS`, `MEMORY_DECAY_MIN_PRIORITY`, `MEMORY_DECAY_ENABLED` trong [.env.example](.env.example).

Kiểm thử API thật được chạy riêng, dùng dữ liệu giả và tối đa 10 lời gọi nếu tất cả kiểm tra đạt:

```bash
python src/live_smoke.py --provider gemini --model gemini-3.1-flash-lite --output results/live_gemini.json
```

Lệnh cần `GEMINI_API_KEY`, không đổi `.env`, không chấp nhận trả lời offline thay cho API. Mỗi request giới hạn 256 output token, timeout 20 giây và không retry. Kết quả đã chạy: **10/10 đạt**, lưu câu trả lời, thời gian và token do provider báo trong [live_gemini.json](results/live_gemini.json). Các provider còn lại chưa được gọi API thật.

Mỗi lần benchmark tự làm sạch thư mục riêng của dataset trong `state/benchmarks/`. Profile ở `state/profiles/` được giữ nguyên. Sau lần chạy, có thể mở `state/benchmarks/conversations/profiles/dungct/User.md` hoặc `state/benchmarks/advanced_long_context/profiles/dungct_stress/User.md` để xem fact đã lưu.

Benchmark cần in ra hai bảng: **Standard Benchmark** và **Long-Context Stress Benchmark**. Mỗi bảng so sánh Baseline với Advanced theo đủ 6 cột trong phần "Chỉ số benchmark cần hiểu".

## Cách dùng repo này

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.
