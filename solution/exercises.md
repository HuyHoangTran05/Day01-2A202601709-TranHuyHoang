# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay từng dòng giữ chỗ bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi chạy thật, temperature 0.0 và 0.5 đều đưa ra 7 sự thật, chủ yếu xoay quanh Sơn Đoòng, xe máy, cà phê, họ Nguyễn và múa rối nước; temperature 1.0 đưa ra 8 mục và thêm văn hóa ẩm thực đường phố, còn temperature 1.5 đưa ra 9 mục, thêm sáu thanh điệu tiếng Việt và Cầu Vàng. Nhìn chung, temperature càng cao thì phản hồi càng dài, nhiều ý mới và cách trình bày giàu cảm xúc hơn, trong khi các mức thấp lặp lại các thông tin cốt lõi và có cấu trúc ổn định hơn. Tuy nhiên, vì mỗi mức chỉ được gọi một lần nên kết quả này mới thể hiện xu hướng, chưa đủ để kết luận tuyệt đối về độ ngẫu nhiên.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Theo em nên đặt temperature dưới 0.3 để chatbot trả lời nhất quán, tránh hallucination hay tự bịa thêm nội dung. Mức này giúp chatbot bám sát câu hỏi và chính sách hỗ trợ khách hàng, đồng thời vẫn đủ linh hoạt để diễn đạt tự nhiên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 × 3 × 350 = 10,5 triệu output token mỗi ngày, theo bảng giá trong bài thì GPT-4o tốn khoảng 105 USD/ngày còn GPT-4o-mini khoảng 6,3 USD/ngày, nên GPT-4o đắt hơn khoảng 16,7 lần nếu chỉ xét output token. Theo em GPT-4o phù hợp với tác vụ phức tạp như phân tích hợp đồng hoặc xử lý yêu cầu khó, còn GPT-4o-mini phù hợp với FAQ, phân loại câu hỏi và các phản hồi đơn giản có số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, model trả lời ngắn trong 4 câu, dùng hình ảnh “cuốn sổ nhật ký chung kỳ diệu” của cả lớp để giải thích nên trẻ em dễ hình dung. Với persona chuyên gia tài chính, phản hồi dài và dùng nhiều thuật ngữ như sổ cái phân tán (DLT), hàm băm mật mã, cơ chế đồng thuận, Proof of Work, Proof of Stake và rủi ro đối tác. Phản hồi thứ hai cũng tập trung vào ứng dụng tài chính như hợp đồng thông minh và thanh toán xuyên biên giới, thay vì ví dụ lớp học. Như vậy, system prompt đã thay đổi rõ đối tượng người đọc, giọng điệu, độ dài, vốn từ và mức độ chuyên sâu của câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Em thử một đoạn tiếng Việt dài 114 từ. Công thức `số từ / 0.75` ước lượng 152 token, còn `count_tokens` với tokenizer của GPT-4o đếm được 277 token; kết quả thực tế cao hơn 125 token, tương đương khoảng 82,2% so với số ước lượng. Tiếng Việt thường tốn nhiều token vì từ có dấu và các âm tiết tiếng Việt không phải lúc nào cũng có sẵn như một token hoàn chỉnh trong bộ từ vựng, nên tokenizer phải tách chúng thành nhiều token con; bộ từ vựng cũng thường được tối ưu tốt hơn cho các từ tiếng Anh phổ biến.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi model tạo phản hồi dài hoặc có độ trễ cao, chẳng hạn chatbot tư vấn hay trợ lý viết nội dung, vì người dùng có thể đọc phần đầu ngay thay vì nhìn màn hình chờ đến khi toàn bộ câu trả lời hoàn tất. Non-streaming phù hợp hơn với phản hồi ngắn, tác vụ chạy nền, hoặc khi ứng dụng cần nhận đủ kết quả để kiểm tra định dạng, kiểm duyệt nội dung hay parse một cấu trúc JSON trước khi hiển thị và xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, nhờ đó giảm số request gửi tới API đang quá tải và cho dịch vụ thêm thời gian phục hồi. Nếu hàng nghìn client đều retry sau một delay cố định giống nhau, chúng sẽ gửi request lại gần như cùng lúc, tạo thành các đợt tải lớn lặp lại và có thể khiến API tiếp tục nghẽn. Trong thực tế nên kết hợp exponential backoff với jitter ngẫu nhiên để các client không retry đồng thời.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” Cụm “trợ giảng thân thiện” định hướng model giải thích theo cách hỗ trợ, dễ tiếp cận thay vì quá hàn lâm. Yêu cầu “trả lời ngắn gọn bằng tiếng Việt” giúp phản hồi đúng ngôn ngữ của lớp học, đi thẳng vào trọng tâm và tránh làm người học bị ngợp bởi thông tin không cần thiết.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là chỉ giữ 3 lượt hội thoại gần nhất (`history[-6:]`), nên có thể quên mục tiêu hoặc thông tin quan trọng mà người dùng đã nêu từ đầu phiên. Em sẽ cải thiện bằng cách giữ nguyên 3 lượt gần nhất nhưng tóm tắt các lượt cũ thành một message ngắn; mỗi khi history vượt giới hạn, chương trình gọi model tạo bản tóm tắt, lưu bản đó và chèn nó sau system prompt trong những lần gọi tiếp theo.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
