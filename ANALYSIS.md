# Phân tích Lab 17 — Memory Systems for AI Agent

## 1. Cách chạy và phạm vi phép đo

Bài làm hoàn thiện cấu hình, sáu provider, memory store, Baseline, Advanced, hai benchmark và bộ kiểm thử. Bonus gồm **Conflict handling**, **Confidence threshold** và **Memory decay**. Luồng live đã được kiểm tra với API Gemini thật, tách riêng khỏi benchmark offline.

Kết quả chính được đo ở chế độ **offline**, không gọi API, trên hai file dữ liệu gốc không chỉnh sửa:

- Standard: 10 hội thoại, 101 lượt người dùng và 14 câu hỏi recall.
- Stress: 1 hội thoại, 16 lượt dài và 3 câu hỏi recall.

Mỗi câu hỏi recall được hỏi trong một thread mới, ngay sau hội thoại tương ứng. Hai agent nhận cùng input, cùng thứ tự và cùng câu hỏi. Câu trả lời dựa trên memory, không được cung cấp `expected_contains`; trường này chỉ dùng để chấm điểm sau khi agent trả lời.

Chạy từ thư mục gốc, sau khi kích hoạt `.venv` và cài `requirements.txt`:

```bash
python src/benchmark.py --output results/benchmark.md --json results/benchmark.json
pytest src/test_agents.py -v
python src/benchmark.py --no-compact --output results/benchmark_no_compact.md --json results/benchmark_no_compact.json
python src/bonus_benchmark.py
python src/bonus_policies.py
```

Benchmark tự làm sạch thư mục của mỗi dataset trong `state/benchmarks/`, không xóa profile sử dụng riêng ở `state/profiles/`. Hai lần chạy lại trên workspace benchmark sạch cho cùng số liệu. `pytest.ini` đặt thư mục tạm riêng cho test tại `state/pytest` để tránh lỗi quyền truy cập thư mục Temp của Windows.

## 2. Quy ước cấu hình

`load_config()` nạp `.env` ở gốc repo, giữ ưu tiên cho biến môi trường đã export và tạo `state/`. Model chính và model judge có cấu hình riêng; các benchmark trong bài dùng heuristic, không gọi judge model.

| Biến | Ý nghĩa / mặc định |
| --- | --- |
| `LLM_MODE`, `JUDGE_MODE` | `offline`; chỉ bật `live` khi muốn gọi model thực |
| `LLM_PROVIDER`, `JUDGE_PROVIDER` | Provider chính mặc định `openai`; judge mặc định theo provider chính |
| `LLM_MODEL`, `JUDGE_MODEL` | `stub`; chế độ live yêu cầu tên model thực |
| `LLM_TEMPERATURE`, `JUDGE_TEMPERATURE` | `0` |
| `COMPACT_THRESHOLD_TOKENS` | Mặc định trong code và `.env.example`: `1200` |
| `COMPACT_KEEP_MESSAGES` | `4` message, gồm cả user và assistant |
| `MEMORY_CONFIDENCE_THRESHOLD` | `0.8`; chỉ ghi candidate có điểm đạt ngưỡng |
| `MEMORY_DECAY_HALF_LIFE_TURNS` | `100`; số lượt của user để priority giảm một nửa |
| `MEMORY_DECAY_MIN_PRIORITY` | `0.2`; fact dưới ngưỡng không được đưa vào phần profile của prompt |
| `MEMORY_DECAY_ENABLED` | `true`; có thể tắt decay bằng `false` |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | Key cho provider tương ứng |
| `CUSTOM_API_KEY`, `CUSTOM_BASE_URL` | Provider tương thích OpenAI; bắt buộc có base URL khi dựng model |
| `OLLAMA_BASE_URL` | Địa chỉ Ollama; không cần API key |
| `LLM_API_KEY`, `JUDGE_API_KEY`, `LLM_BASE_URL`, `JUDGE_BASE_URL` | Giá trị riêng cho từng model, ưu tiên hơn biến của provider |

`.env` hiện tại đặt ngưỡng **1800 token**, giữ **4 message**, chọn Gemini nhưng vẫn ở chế độ offline. Các bảng chính dưới đây phản ánh cấu hình này. Phép thử ngưỡng mặc định 1200 được lưu riêng ở [benchmark_threshold1200.md](results/benchmark_threshold1200.md). Không ghi key vào tài liệu hoặc kết quả.

Sáu adapter được kiểm tra bằng cách khởi tạo model với key giả mà không gọi API. Provider SDK được import theo nhu cầu. Tham số model được đối chiếu với [LangChain Reference](https://reference.langchain.com/python/langchain-openrouter/chat_models/ChatOpenRouter).

## 3. Định nghĩa sáu chỉ số

| Chỉ số | Cách đo |
| --- | --- |
| Agent tokens only | Tổng token ước lượng trong câu trả lời, gồm cả lượt chat và lượt recall |
| Prompt tokens processed | Tổng token ước lượng của ngữ cảnh đầu vào ở từng lượt; cùng system prompt cho hai agent |
| Cross-session recall | Mỗi câu hỏi: `1` nếu đủ chuỗi mong đợi, `0.5` nếu có một phần, `0` nếu không có; lấy trung bình các câu hỏi |
| Response quality | `0.8 × tỷ lệ chuỗi đúng + 0.2 × điều kiện câu trả lời không rỗng và ≤ 600 ký tự`; lấy trung bình các câu hỏi recall |
| Memory growth (bytes) | Tổng kích thước `User.md` và `Memory.json` cuối trừ đầu, theo byte UTF-8, trên các user của dataset |
| Compactions | Tổng số lần nén thực sự giảm ngữ cảnh, gồm các thread chat và recall |

`estimate_tokens()` trả `max(1, len(text.strip()) // 4)` cho chuỗi không rỗng và `0` cho chuỗi rỗng. Đây là heuristic ký tự, không phải tokenizer hoặc hóa đơn của provider. Cả hai agent dùng cùng estimator và cùng chính sách trả lời offline.

Baseline mang theo toàn bộ lịch sử của thread. Advanced mang theo system prompt, các fact còn đủ priority từ `User.md`, summary và các message gần nhất. `Memory.json` chỉ dùng tính priority, không được chèn vào prompt. Compact được thực hiện trước khi dựng prompt; tóm tắt dùng Python nên không có chi phí LLM riêng trong phép đo. Nếu thay bằng LLM summarizer, cần cộng thêm chi phí của lần tóm tắt.

## 4. Kết quả chính

Số liệu đầy đủ ở [benchmark.md](results/benchmark.md) và [benchmark.json](results/benchmark.json).

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Recall | Quality | Memory growth (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 1846 | 21642 | 0.000 | 0.200 | 0 | 0 |
| Standard | Advanced | 2049 | 29504 | 1.000 | 1.000 | 1215 | 0 |
| Stress | Baseline | 337 | 23695 | 0.000 | 0.200 | 0 | 0 |
| Stress | Advanced | 394 | 19091 | 1.000 | 1.000 | 993 | 1 |

### Vì sao Advanced nhớ qua phiên tốt hơn?

Recall của Baseline bằng 0 ở cả hai bộ, trong khi Advanced bằng 1. Baseline khóa `SessionState` theo `thread_id`; thread recall mới không có lịch sử cũ và không đọc profile. Advanced trích fact từ câu khai báo, ghi vào `User.md` theo `user_id` rồi đọc lại trong thread mới. Test cũng xác nhận profile tồn tại sau khi tạo một instance Advanced mới, không chỉ nhờ một dict còn sống trong RAM.

Đây là kết quả trên hai bộ dữ liệu cụ thể, không chứng minh agent hiểu mọi cách diễn đạt tiếng Việt. Recall chỉ kiểm tra chuỗi con; một câu chứa chuỗi đúng nhưng diễn giải sai vẫn có thể được chấm cao. Quality cũng là heuristic phụ thuộc fact coverage, không phải đánh giá độc lập về chất lượng ngôn ngữ.

### Vì sao Advanced tốn hơn ở hội thoại ngắn?

Ở Standard, Advanced xử lý 29.504 prompt token so với 21.642 của Baseline, tăng **36,3%**. Cột Compactions bằng 0: hội thoại ngắn chưa vượt ngưỡng, nên profile thêm vào mỗi prompt tạo chi phí mà compact chưa bù lại. Advanced còn trả lời được các câu recall bằng thông tin cụ thể, dẫn đến 2.049 output token so với 1.846 của Baseline.

Việc ghi file không trực tiếp tạo token. Profile xuất hiện trong đầu vào làm tăng **Prompt tokens processed**; câu trả lời được sinh ra mới đóng góp vào **Agent tokens only**. Hai agent có output token bằng nhau cũng không đủ để kết luận profile không được ghi.

### Vì sao compact có lợi thế ở hội thoại dài?

Stress có ít lượt hơn Standard nhưng Baseline vẫn xử lý 23.695 prompt token, do các lượt dài và lịch sử cũ được mang theo nhiều lần. Advanced xử lý 19.091 prompt token, giảm **19,4%** so với Baseline, đồng thời vẫn đạt recall 1. Cột Compactions bằng 1 xác nhận nén đã chạy.

Ngữ cảnh ở một lượt có xu hướng tăng theo độ dài lịch sử. Tổng prompt token cộng dồn qua nhiều lượt có thể tăng gần bậc hai theo số lượt nếu độ dài từng lượt tương đương và giữ toàn bộ lịch sử; không nên nhầm tổng này với kích thước prompt ở lượt cuối.

## 5. Phép thử tắt compact và thay ngưỡng

Kết quả đầy đủ ở [benchmark_no_compact.md](results/benchmark_no_compact.md).

| Advanced trên Stress | Prompt tokens processed | Agent tokens only | Recall | Compactions |
| --- | ---: | ---: | ---: | ---: |
| Tắt compact, ngưỡng `10**12` | 25101 | 394 | 1.000 | 0 |
| Ngưỡng 1800, cấu hình `.env` hiện tại | 19091 | 394 | 1.000 | 1 |
| Ngưỡng 1200, mặc định trong code | 14672 | 394 | 1.000 | 3 |

So với chính Advanced khi tắt compact, cấu hình 1800 giảm **23,9% prompt token**. Output token không đổi ở cả ba trường hợp. Điều này tách được lợi ích của lớp compact khỏi lợi ích recall của persistent memory. Tắt compact khiến Advanced còn tốn ngữ cảnh hơn Baseline vì vẫn phải chèn profile.

Ngưỡng 1200 làm compact chạy ba lần, giảm prompt token hơn. Đây chưa đủ để kết luận nên luôn giảm ngưỡng: recall ở bộ này chỉ hỏi fact trong `User.md`, không kiểm tra đầy đủ các chi tiết tin tức tạm thời. Nén mạnh hơn có thể làm mất chi tiết cần cho follow-up. Sau các phép thử, cấu hình chính vẫn đọc `.env` như trước.

## 6. Memory growth và rủi ro

Baseline tăng 0 byte vì không tạo profile. Advanced tăng 1.215 byte ở Standard và 993 byte ở Stress, gồm metadata. Riêng `User.md` vẫn là 347 và 435 byte; `Memory.json` thêm 868 và 558 byte. Khi dữ liệu mới trùng khóa, `upsert_fact()` sửa dòng hiện có thay vì nối thêm fact mâu thuẫn; khi giá trị không đổi, không ghi lại `User.md`. Metadata vẫn cập nhật mỗi lượt để giữ tuổi của fact qua restart. Preference về style được hợp nhất, số bullet mới thay thế số cũ; yêu cầu không dùng bullet hoặc trả lời dài loại bỏ preference đối lập.

Các file có thể kiểm tra sau benchmark:

- `state/benchmarks/conversations/profiles/dungct/User.md`
- `state/benchmarks/advanced_long_context/profiles/dungct_stress/User.md`

Memory growth đo hai file persistent memory, không đo RAM chứa lịch sử hoặc summary. Kích thước file cũng không thể hiện tổng số byte đã ghi qua mọi lượt. Metadata làm tăng dung lượng lưu trữ; decay hiện tại chỉ giảm fact đưa vào prompt, không xóa dữ liệu gốc hay giảm dung lượng file.

Regex chỉ nhận diện một số mẫu khai báo và có thể bỏ sót cách diễn đạt lạ hoặc phân loại sai sở thích. Summary giữ fact cùng excerpt ngắn, có giới hạn kích thước; đây không phải tóm tắt ngữ nghĩa đáng tin cậy cho mọi tin tức. Một message mới rất dài hoặc các message được yêu cầu giữ nguyên có thể tự vượt ngưỡng, nên ngưỡng là điều kiện kích hoạt nén, không phải giới hạn cứng cho mọi prompt.

Profile có thể chứa thông tin riêng tư và cần cơ chế sửa/xóa khi dùng thực tế. Path được chuẩn hóa và kiểm tra nằm trong root, user được cách ly, và thread không được dùng chung giữa hai user. Mỗi file được ghi bằng file tạm rồi thay thế, nhưng hai file chưa có transaction chung hoặc khóa cho nhiều tiến trình đồng thời. Quy trình hỏi người dùng xác nhận correction chưa được triển khai.

## 7. Bonus: xử lý correction

Bonus giải quyết việc nơi ở và nghề nghiệp cũ tiếp tục ảnh hưởng recall. Mỗi fact có một khóa rõ ràng; `edit_text()` thay giá trị cũ. Câu hỏi, giả định, câu đùa hoặc thông tin đi họp không được coi là correction. Không để cùng lúc hai dòng `location` hoặc hai dòng `profession` trong profile do agent tạo.

`bonus_benchmark.py` tạo một đối chứng riêng: **First fact wins** khóa nơi ở/nghề nghiệp sau lần ghi đầu, còn mọi phần khác giữ cùng chính sách với Advanced. Mỗi variant chạy trên workspace sạch, cùng input và estimator. Số liệu ở [bonus_conflicts.md](results/bonus_conflicts.md) và [bonus_conflicts.json](results/bonus_conflicts.json).

| Bộ dữ liệu | Recall khi giữ fact đầu | Recall khi nhận correction | Prompt token trước → sau | Memory byte trước → sau |
| --- | ---: | ---: | --- | --- |
| Standard | 0.643 | 1.000 | 29615 → 29504 | 1221 → 1215 |
| Stress | 0.667 | 1.000 | 19080 → 19091 | 986 → 993 |

Bonus cải thiện recall ở cả hai bộ. Nó không bảo đảm giảm token hoặc byte: ở Stress, prompt tăng 11 token, `User.md` tăng 6 byte và tổng persistent memory tăng 7 byte. Mục tiêu chính là lưu đúng fact hiện tại.

Rủi ro tăng thêm là một khai báo mới bị hiểu sai có thể ghi đè fact đúng trước đó. Confidence threshold giúp lọc một số mẫu nhiễu, nhưng chưa có lịch sử provenance đầy đủ hoặc xác nhận từ người dùng. Correction dạng “Mình đã chuyển từ Huế sang Hà Nội.” và rút lại preference bullet đã có test riêng.

## 8. Bonus: confidence threshold

`extract_profile_candidates()` gắn điểm bằng quy tắc cho từng candidate: khai báo trực tiếp `0.98`, câu không chắc chắn `0.45`, khai báo được trích dẫn `0.25`, thông tin về người khác `0.15`. `apply_candidates()` chỉ ghi khi điểm ≥ ngưỡng cấu hình, mặc định `0.8`. Candidate bị từ chối không ghi đè giá trị cũ và không làm mới tuổi của fact. Câu hỏi recall cũng không được coi là xác nhận lại.

Đây là **điểm bằng quy tắc, không phải xác suất đã hiệu chỉnh**. Regex chỉ bao phủ một số mẫu; câu có cả thông tin người dùng và người khác có thể bị lọc quá mức vì điểm tính theo câu. Threshold cao có thể bỏ sót fact đúng, thấp có thể làm nhiễu hồ sơ.

Phép thử synthetic trong [bonus_policies.md](results/bonus_policies.md) và [bonus_policies.json](results/bonus_policies.json) dùng cùng input: khai báo Lan/Huế, thông tin bạn tên Nam, rồi “Có lẽ mình đang ở Hà Nội.”. Tắt decay và compact trong cả hai variant để tách tác động của threshold.

| Variant | Recall | Quality | Prompt token | Memory byte |
| --- | ---: | ---: | ---: | ---: |
| Threshold 0 | 0.000 | 0.200 | 350 | 294 |
| Threshold 0.8 | 1.000 | 1.000 | 348 | 290 |

Threshold `0.8` giữ tên và nơi ở đã khai báo chắc chắn trong ví dụ này. Kết quả trên một tình huống nhỏ chưa chứng minh lọc chính xác mọi câu tiếng Việt.

## 9. Bonus: memory decay

Mỗi user có bộ đếm lượt persistent trong `Memory.json`. Fact giữ `confidence`, `last_seen_turn`, `seen_count`; priority tính bằng `confidence × 2^(-age / half_life)`. Tên không giảm theo tuổi; các fact còn lại chỉ được đưa vào phần profile của prompt khi priority ≥ `min_priority`. Mặc định half-life 100 lượt và ngưỡng `0.2`. Các lượt của user khác không làm tăng tuổi; thời gian không hoạt động ngoài hội thoại cũng không làm giảm priority.

Một khai báo đạt threshold xác nhận lại fact sẽ cập nhật `last_seen_turn` và khôi phục priority. Fact hết ưu tiên vẫn ở `User.md`, không bị xóa. Profile cũ chưa có metadata được khởi tạo confidence `1.0` khi đọc và lưu metadata khi xử lý lượt tiếp theo. Cơ chế áp dụng cho persistent memory; short-term history và summary trong thread có thể vẫn chứa fact cũ.

Phép thử trong `bonus_policies.py` khai báo tên/nơi ở/sở thích, thêm 10 lượt không xác nhận fact, rồi hỏi trong thread mới. Compact tắt ở cả hai variant; half-life cố ý giảm còn 2 lượt và ngưỡng tăng lên `0.3` để thấy tác động.

| Variant | Recall | Quality | Prompt token | Memory byte |
| --- | ---: | ---: | ---: | ---: |
| Decay off | 1.000 | 1.000 | 2509 | 417 |
| Half-life 2 turns | 0.500 | 0.600 | 2437 | 417 |

Decay giảm **2,9% prompt token**, nhưng recall giảm do nơi ở chưa được xác nhận lại bị loại. Dung lượng file giữ nguyên. Tuổi không đủ để suy ra fact đã sai: giảm half-life quá mạnh có thể quên thông tin vẫn còn đúng. Cấu hình mặc định bảo toàn recall 1 trên hai benchmark chính; chưa có nghiên cứu đủ rộng để chọn tham số tối ưu.

## 10. Kiểm thử API live

Chạy riêng, không đổi `.env` hoặc gửi dữ liệu benchmark cá nhân:

```bash
python src/live_smoke.py --provider gemini --model gemini-3.1-flash-lite --output results/live_gemini.json
```

**10/10 kiểm tra đạt, 10 lời gọi API thật** với `gemini-3.1-flash-lite`. [live_gemini.json](results/live_gemini.json) lưu câu trả lời synthetic, thời gian và token thực do provider trả về. Kiểm tra gồm Baseline nhớ trong thread/quên ở thread mới; Advanced nhớ sau restart, nhận correction, không thay tên bằng tên người bạn; decay bỏ nơi ở cũ trong thread mới nhưng vẫn giữ tên.

Mỗi request giới hạn 256 output token, timeout 20 giây, không retry. Script yêu cầu model thật và key phù hợp, không cho phép fallback offline; lỗi chỉ lưu loại exception và status code, không lưu nội dung lỗi provider có thể chứa thông tin nhạy cảm. Thư mục thử tạm nằm trong `state/` và được dọn sau khi chạy. Bộ pytest không gọi API.

Lần thử trước với `gemini-2.5-flash-lite` báo `GoogleModelNotFoundError`; lưu ở [live_gemini_model_unavailable.json](results/live_gemini_model_unavailable.json), không tính vào 10 kiểm tra đạt. Khả năng dùng model phụ thuộc tài khoản/endpoint; có thể xem model theo [Google Models API](https://ai.google.dev/api/models). Các provider còn lại chưa được gọi API thật. Live smoke không thay thế benchmark chất lượng LLM trên toàn bộ dataset.

## 11. Kiểm chứng

Bộ `src/test_agents.py` có **48 trường hợp pass**, gồm bốn test bắt buộc và các test bổ sung cho:

- Correction, câu hỏi không có dấu `?`, giả định và thông tin nhiễu.
- Path traversal, tên thư mục đặc biệt trên Windows và va chạm user ID.
- Recall sau khi tạo lại agent, cách ly user và ngữ cảnh tạm chỉ sống trong thread.
- Cộng dồn token theo lượt, giới hạn summary và compact giảm prompt load.
- Alias/provider, `.env`, offline không dựng model và nhánh live qua model giả.
- Dataset thật, benchmark tái lập và phép thử tắt compact.
- Confidence ở biên threshold, candidate yếu không ghi đè hoặc refresh, khai báo được trích dẫn/thông tin người khác.
- Decay đúng half-life, expiry, xác nhận lại, restart, migration profile cũ và cấu hình env.
- Rút lại preference; script live từ chối fallback offline và không ghi nội dung lỗi provider.

`pip check` không phát hiện dependency bị hỏng. Hai dataset trong `data/` giữ nguyên. Hai benchmark chính và các phép thử bonus đều chạy offline; kiểm tra Gemini live lưu riêng. Không thực hiện commit hoặc push.
