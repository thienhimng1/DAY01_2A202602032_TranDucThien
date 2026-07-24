# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng placeholder câu trả lời mẫu bằng
nội dung thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0, model gần như trả lời giống hệt nhau ở mọi lần gọi lại
> (deterministic) — thường chọn một sự thật "an toàn", phổ biến (ví dụ Vịnh
> Hạ Long, phở). Từ 0.5 lên 1.0, câu chữ và sự thật được chọn bắt đầu đa dạng
> hơn, đôi khi sáng tạo cách diễn đạt hơn. Ở 1.5, phản hồi dễ trở nên lan
> man, đôi khi lặp từ kỳ lạ hoặc pha trộn thông tin thiếu chính xác — dấu
> hiệu model đang "đoán" quá tự do so với phân phối xác suất gốc.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần câu trả lời nhất quán, đúng
> chính sách công ty và ít rủi ro "bịa" thông tin (hallucination) — temperature
> thấp giúp model bám sát nội dung có xác suất cao nhất thay vì thử các cách
> diễn đạt "sáng tạo" không cần thiết. Không đặt 0.0 tuyệt đối vì vẫn muốn
> câu trả lời nghe tự nhiên, không bị máy móc khi lặp lại cùng một câu hỏi.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Giá output GPT-4o là $0.010/1K token, GPT-4o-mini là $0.0006/1K token →
> GPT-4o đắt hơn đúng **16,67 lần** trên mỗi token (0.010 / 0.0006). Với
> workload này: tổng output = 10.000 × 3 × 350 = 10.500.000 token/ngày →
> GPT-4o ≈ 10.500 × 0.010 = **$105/ngày**, mini ≈ 10.500 × 0.0006 =
> **$6.30/ngày** (chênh lệch **$98.70/ngày**, tức ~$36.000/năm). GPT-4o đáng
> dùng cho tác vụ cần lập luận phức tạp/độ chính xác cao (ví dụ tư vấn pháp
> lý, debug code); mini phù hợp cho việc đơn giản, khối lượng lớn như phân
> loại intent, trả lời FAQ, tóm tắt ngắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Bản "giáo viên tiểu học" thường ngắn hơn, dùng ví dụ đời thường (như "cuốn
> sổ ghi chép mà ai cũng có bản sao"), tránh hoàn toàn thuật ngữ kỹ thuật.
> Bản "chuyên gia tài chính" dài hơn, dùng từ như "sổ cái phân tán",
> "consensus", "immutability", có thể nhắc đến ứng dụng trong DeFi hoặc tài
> sản số. Cùng một câu hỏi nhưng system prompt định hình toàn bộ giọng văn,
> độ sâu kiến thức giả định và mức độ thuật ngữ — chứng tỏ system prompt
> không chỉ là "ngữ cảnh" mà thực sự lái hành vi và persona của model xuyên
> suốt cả câu trả lời, mạnh hơn nhiều so với chỉ thêm yêu cầu vào user prompt.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn ~100 từ tiếng Việt, `count_tokens` (tiktoken, encoding
> `cl100k_base`) thường trả về khoảng 180–220 token, trong khi công thức
> `số từ / 0.75` (vốn hiệu chỉnh cho tiếng Anh) chỉ ước lượng ~133 token —
> chênh lệch khoảng 40–60%, tức công thức ước lượng đang **đánh giá thấp**
> chi phí thực tế cho tiếng Việt. Nguyên nhân: bộ mã hoá BPE của OpenAI được
> huấn luyện chủ yếu trên dữ liệu tiếng Anh, nên mỗi âm tiết tiếng Việt có
> dấu (ví dụ "nguyên", "đường") thường bị tách thành 2–3 sub-token thay vì 1,
> do các ký tự có dấu (ư, ơ, ệ, ...) không nằm trong các cụm byte phổ biến
> mà BPE đã học — khiến tiếng Việt "tốn" token hơn hẳn dù số từ bằng nhau.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi con người trực tiếp chờ đọc phản hồi dài
> (chatbot, trợ lý viết nội dung) — nó giảm cảm giác chờ đợi vì người dùng
> thấy chữ xuất hiện ngay từ giây đầu tiên (time-to-first-token thấp) thay
> vì nhìn màn hình trắng vài giây rồi cả đoạn văn bật ra cùng lúc, dù tổng
> thời gian xử lý thực tế không đổi. Ngược lại, non-streaming phù hợp hơn khi
> output cần được xử lý tiếp bởi code trước khi hiển thị — ví dụ parse JSON
> có cấu trúc, gọi function calling, hoặc chạy trong pipeline batch không có
> người dùng xem trực tiếp — vì lúc đó ta cần response hoàn chỉnh để validate
> hoặc parse, streaming từng phần chỉ làm code phức tạp thêm mà không mang
> lại lợi ích trải nghiệm nào.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ (0.1s → 0.2s → 0.4s → ...) nên
> cho server thời gian phục hồi nhiều hơn nếu request đầu vẫn thất bại, thay
> vì liên tục "gõ cửa" với cùng tần suất trong lúc server đang quá tải. Nếu
> hàng nghìn client cùng dùng delay cố định giống nhau, tất cả sẽ retry cùng
> lúc theo từng đợt đồng bộ (thundering herd) — mỗi đợt retry lại tạo ra một
> đỉnh tải mới đúng bằng lúc server vừa hồi phục một chút, khiến server không
> bao giờ kịp ổn định và có thể sập hẳn. Trong thực tế nên kết hợp thêm
> "jitter" (độ trễ ngẫu nhiên nhỏ cộng vào mỗi lần chờ) để tránh các client
> retry đồng loạt cùng một mốc thời gian.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: *"Bạn là trợ giảng AI thân thiện của khóa học AI Practical
> Competency Program. Luôn trả lời ngắn gọn (tối đa 3–4 câu) bằng tiếng Việt,
> ưu tiên ví dụ cụ thể thay vì lý thuyết dài dòng. Nếu không chắc chắn về một
> thông tin, hãy nói rõ thay vì đoán."* — Yêu cầu "trả lời ngắn gọn" quan
> trọng vì mỗi lượt trả lời dài sẽ tốn nhiều token output hơn (chi phí tăng
> theo cấp số cộng mỗi lượt) và trong ngữ cảnh trợ giảng, câu trả lời cô đọng
> giúp học viên không bị "ngợp" thông tin. Chỉ định rõ "bằng tiếng Việt" để
> tránh model tự chuyển sang tiếng Anh khi câu hỏi có lẫn thuật ngữ kỹ thuật
> tiếng Anh (rất hay gặp khi hỏi về code/API).

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: history chỉ giữ 3 lượt gần nhất (6 message), nên nếu
> người dùng hỏi lại điều đã nói cách đây 5–6 lượt, trợ lý sẽ "quên" hoàn
> toàn ngữ cảnh đó — không có bộ nhớ dài hạn xuyên suốt phiên. Cải thiện cụ
> thể: thêm một bước tóm tắt (summarization) — mỗi khi history sắp bị cắt,
> gọi thêm một lần API để nén các lượt cũ nhất thành 1–2 câu tóm tắt, lưu vào
> một message `system` phụ ("Bối cảnh trước đó: ..."), rồi mới cắt history
> chi tiết. Cách này giữ được ngữ cảnh quan trọng qua nhiều lượt mà không để
> token input phình to vô hạn như khi lưu toàn bộ lịch sử gốc.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
