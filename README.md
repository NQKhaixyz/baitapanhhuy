# Mini RAG Engine — lõi hỏi–đáp y khoa

Repo này triển khai trọn 5 chặng trên corpus 8 đoạn đã cho. Mục tiêu là làm rõ
lõi RAG và đo được từng bước: không database, không chunking, không frontend,
không FAISS/Chroma/LangChain ở tầng retrieval.

## Cài đặt

Yêu cầu Python 3.11+:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Embedding mặc định là Gemini `gemini-embedding-001`. Repo tự nạp các biến từ file `.env` (file này đã được gitignore):

```powershell
Copy-Item .env.example .env
# Mở .env, điền GEMINI_API_KEY thật rồi chạy:
uv run python .\\eval.py
```

Chặng 1–4 dùng chính key này cho Gemini Generate Content; nếu Gemini không trả
citation hợp lệ, engine sẽ yêu cầu model sửa một lần rồi từ chối an toàn nếu
vẫn không đạt contract. Response hợp lệ được cache theo model + prompt ở
`.cache/generation/`, nên chạy lại `eval.py` không gọi Gemini lặp lại. Nếu model
chính trả 429, adapter tự chuyển một lần sang `GEMINI_FALLBACK_MODEL`.

Nếu chưa có key hoặc cần chạy offline:

```powershell
$env:RAG_EMBEDDING_PROVIDER = "local"
python eval.py
```

Provider local là vector hashing tái lập, chỉ là fallback để kiểm thử luồng; khi
có Gemini, 8 vector corpus được tạo một lần và lưu ở `.cache/corpus_embeddings.npy`,
còn fingerprint nằm trong file `.npy.meta.json` cạnh đó. Xóa cache sau khi đổi
corpus hoặc model. API key thật không nằm trong repo.

## Chạy năm chặng

Mỗi file chạy độc lập, không import chặng sau:

```powershell
python stage0.py --all                         # retrieval + top-k + recall@3
python stage1.py --all                         # câu trả lời grounded + citation
python stage2.py --all                         # validate/router + graph
python stage3.py                               # hội thoại + rewrite
python stage4.py                               # tương đương eval.py
python eval.py                                 # recall@1/@3/MRR + guardrail
```

Chạy một câu thủ công:

```powershell
python stage0.py --query "Phân độ tăng huyết áp gồm mấy độ?"
python stage1.py --query "Bệnh nhân eGFR 20 ml/phút có dùng được metformin không?"
```

## Kiến trúc

```text
rewrite (C3) → validate (C2) → route (C2) → retrieve (C0)
                                                    ↓
                                   synthesize (C1) → cite (C1)
```

- `rag_engine/embeddings.py`: Gemini adapter, local fallback, chuẩn hóa vector và cache.
- `rag_engine/retrieval.py`: tự viết cosine bằng numpy, lọc guideline trước khi tìm.
- `rag_engine/graph.py`: state `TypedDict`, node/conditional routing và log từng bước.
- `rag_engine/synthesis.py`: gọi Gemini generation với grounded prompt,
  threshold từ chối, citation whitelist; không có context thì không bịa.
- `rag_engine/conversation.py`: giữ 3 lượt gần nhất trong RAM, tách câu gốc và
  `standalone_query`.
- `eval.py`: chỉ đọc `golden-dataset`, chấm retrieval theo `chunk_id`, kiểm tra
  câu bẫy, citation hợp lệ và ghi `artifacts/eval_results.json` có timestamp.

## Tiêu chí chấm và bẫy đã xử lý

- Retrieval: recall@1, recall@3, MRR; câu 00003 cần hai chunk nên không được
  chỉ kiểm tra một hit.
- Generation: không gọi model nếu cosine dưới ngưỡng; mọi khẳng định có
  `[chunk_id]` tồn tại trong context. Câu eGFR 20 phải so sánh `20 < 30`, không
  trả lời chung chung.
- Câu 00006 không có chunk vàng: engine phải từ chối và nói rõ guideline không
  nêu liều insulin cố định, không tự bịa con số.
- Validator chặn greeting/off-topic trước retrieval; router giữ guideline THA
  và ĐTĐ tách biệt. Log cho biết danh sách chunk còn lại sau route.

## Báo cáo mẫu cần ghi khi nộp

Sau mỗi lần chạy `eval.py`, giữ lại `artifacts/eval_results.json` và ghi trong
thay đổi:

1. recall@3/MRR đã đạt bao nhiêu;
2. một lỗi đã gặp, log nào giúp khoanh vùng (router, threshold hay embedding);
3. một câu engine từng trả lời sai nhưng vẫn tự tin, cùng nguyên nhân và cách
   thêm guardrail.

`golden-dataset/*.jsonl` là dữ liệu bất biến, không sửa đáp án để làm đẹp điểm.

## Kiểm tra hiện tại

- `pytest`: 3 test smoke pass; retrieval stage0 local đạt `recall@3 = 1.0000`
  trên 5 câu có gold chunk.
- Generation chặng 1–4 dùng Gemini thật và cần `GEMINI_API_KEY`; hãy chạy
  `python eval.py` sau khi đặt key để tạo report mới, không dùng lại artifact cũ.
- Lỗi từng gặp: kiểm tra greeting bằng chuỗi con `"hi"` bắt nhầm `"nghi"`/`"nhiêu"`;
  log validate đã giúp phát hiện và đã đổi sang kiểm tra token greeting chính xác.
- Ca dễ tự tin sai: câu insulin từng có hit `dtd2020_ch5_s5.2` score 0.2201 dù
  context không nói insulin. Guardrail theo thực thể + refusal bắt buộc chặn
  việc gọi Gemini và chặn bịa liều.
