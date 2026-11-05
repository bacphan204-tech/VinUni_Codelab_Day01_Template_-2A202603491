# Checkpoint Lab 01

- **Setup:** Clone repo, tạo `.venv`, cài `requirements.txt`.
- **Task 1–3:** Em hỏi ChatGPT cách gọi ba SDK rồi tự sửa `template.py` cho đúng tuple kết quả. Chạy test từng hàm OpenAI, Gemini, Anthropic.
- **Task 4:** Gom kết quả ba model, tính chi phí theo số token và giá trên một triệu token.
- **Task 5:** Test fail vì còn `NotImplementedError`. Em hoàn thiện vòng chat, xử lý `quit/exit` và giữ 6 tin nhắn gần nhất.
- **Bonus A–C:** Thêm retry với thời gian chờ tăng dần, so sánh nhiều prompt và in bảng. B/C fail mấy lần. Đọc traceback mới thấy mock `_get_mock()` không nhận tham số nên em sửa mock ở bản test local. Sau sửa, **9/9 test pass**. Cần rà lại file test trước khi nộp.
- **Gemini API thật:** Test mock pass nhưng gọi thật báo `API_KEY_INVALID`. Em kiểm tra độ dài biến môi trường, copy lại key mới rồi thử riêng bằng một lệnh ngắn. Sau khi gặp lỗi 404 và 503 với hai model khác, `gemini-3.5-flash-lite` trả về `OK`. OpenAI và Anthropic mới qua test mock.
