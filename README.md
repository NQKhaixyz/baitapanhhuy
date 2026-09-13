# Mini RAG Engine — hỏi đáp trên corpus guideline

Engine Python cho 8 chunk và 6 câu hỏi golden: tự tính cosine bằng NumPy,
trả lời có citation, phân loại/route guideline và hội thoại trong RAM.
Không database, chunking, PDF ingestion, frontend hoặc vector database.

## Cài đặt và chạy

Python 3.11+. Môi trường kiểm chứng hiện tại dùng Python 3.14.7.

```powershell
uv sync --extra dev --locked
Copy-Item .env.example .env
# Điền GEMINI_API_KEY trong .env, không commit key.
uv run python stage0.py --all
uv run python stage1.py --all
uv run python stage2.py --all
uv run python stage3.py
uv run python stage4.py
uv run python eval.py --write-review-template
uv run python -m pytest -q
```

Nếu không dùng uv: tạo venv, `pip install -r requirements.txt`, rồi
`pip install pytest` để chạy kiểm thử. Chỉ copy `.env.example` khi chưa có `.env`.
Không cần xóa cache khi đổi nội dung corpus/model; fingerprint tự làm cache cũ
mất hiệu lực. `RAG_EMBEDDING_CACHE` cho phép đặt đường dẫn cache riêng khi chạy
nhiều phép thử.

Embedding Gemini dùng `gemini-embedding-001`; generation dùng model cấu hình
trong `.env`. Corpus được encode theo batch 8 document, lưu `.npy` chuẩn hóa;
query dùng task type retrieval query. Generation cache phân biệt model và toàn
bộ prompt; các response đọc lại vẫn phải qua hậu kiểm hiện tại.

Các helper dùng chung đã được gộp vào `rag_engine/core.py` để repo ít file hơn.
Validator có classifier ba chế độ: `rules` (nhanh, tái lập), `hybrid` (mặc định
khi có key: rule cho ca rõ, LLM Gemini cho ca mơ hồ), và `llm` (gọi LLM cho
mọi câu không rỗng). LLM chỉ trả một nhãn `greeting`/`medical`/`off_topic`,
có cache và luôn fallback về rule khi lỗi. Đặt `RAG_CLASSIFIER=rules` nếu cần
chạy hoàn toàn xác định; đặt `RAG_CLASSIFIER=llm` khi muốn ép dùng classifier.

Chạy offline thật, kể cả máy đã có API key:

```powershell
$env:RAG_OFFLINE = "1"
$env:RAG_RETRIEVAL_THRESHOLD = "0.18"
uv run python stage0.py --all
uv run python eval.py --threshold 0.18
# Khi muốn gọi API trở lại, mở terminal mới hoặc bỏ hai biến override trên.
```

Offline dùng local hashing. Khi không có generation, engine đánh dấu `fallback`,
`rejected=True`, nêu chưa có câu trả lời và chỉ đưa đoạn tham khảo có citation.
Đó không phải câu trả lời đã được kiểm chứng, không được tính là đạt generation.
Để đánh giá đầy đủ C1–C4 cần chạy Gemini và đọc câu trả lời thực tế.

## Năm chặng và sáu node

```text
C3 rewrite → C2 validate → C2 route → C0 retrieve → C1 synthesize → C1 cite
                    └─ greeting / off_topic ───────────────────────→ END
```

- C0/C1 không import LangGraph hoặc node hội thoại; có kiểm thử chặn import
  chặng sau. `search_pairs(query, k)` trả `(chunk_id, score)`; `search` trả thêm
  chunk để dựng context. Vector được chuẩn hóa trước cosine.
- C2 dùng LangGraph thật, State là TypedDict và mỗi node trả phần cập nhật.
  C2 không bật hoặc import rewrite; các câu golden chạy độc lập.
- C3 chỉ chạy rewrite khi có lịch sử. Memory giữ ba lượt với câu gốc,
  standalone query và chủ đề đã giải quyết. Tên bệnh/tên thuốc có trong metadata
  được ưu tiên để nhận biết đổi chủ đề, dù câu bắt đầu bằng “Còn”.
- Greeting có câu hỏi phía sau vẫn được xử lý như câu hỏi. Validator tách chủ đề
  y khoa khỏi việc corpus có đáp án; câu y khoa ngoài corpus được từ chối vì
  thiếu nguồn, không tự động gọi là lạc đề.
- Phân loại không giao toàn bộ cho LLM vì đây là control plane: greeting và ca
  rõ phải dừng nhanh, không tốn lượt gọi và không phụ thuộc model. Chế độ
  `hybrid` vẫn dùng LLM cho câu mơ hồ để nhận ra câu hỏi y khoa ngoài corpus,
  nhưng lỗi mạng/quota không làm engine sập.
- Router ưu tiên chủ đề/guideline được nhắc rõ; có thể giữ nhiều guideline.
  Khi không có chủ đề rõ, dùng điểm cosine theo guideline. Lọc candidate trước
  khi tính top-k; log có toàn bộ `candidates_before_search`, rồi ID và score.
- `synthesize` chỉ gọi model khi vượt ngưỡng relevance. Model luôn được phép
  từ chối vì relevance không chứng minh tài liệu có đáp án. Node `cite` hậu kiểm
  ID thuộc context; không tự gắn nguồn vào một khẳng định không có citation.

Các câu mẫu có thể chạy bằng `stage3.py --query ... --query ...`; mặc định script
chạy THA → phân độ → metformin → chống chỉ định. Mọi node log đầu vào/đầu ra;
lượt đầu log rõ rewrite bị bỏ qua, còn câu gốc dùng để diễn đạt câu trả lời.

## Điểm retrieval và cách hiểu

Bản sửa đạt recall@3 = **1.0000**, MRR = **1.0000** trên năm câu có gold retrieval.
Câu 00006 không có gold, được chấm từ chối riêng, không đưa vào mẫu số recall.

| Chỉ số | k=3 | Khi engine chỉ lấy k=1 |
|---|---:|---:|
| Recall theo tỷ lệ chunk tại 1 | 0.9000 | 0.9000 |
| Recall theo tỷ lệ chunk tại 3 | 1.0000 | 0.9000 |
| Tỷ lệ câu lấy đủ mọi nguồn tại 3 | 1.0000 | 0.8000 |
| MRR | 1.0000 | 1.0000 |

`recall_at_k` là trung bình `số gold chunk lấy được / số gold chunk` trên từng
câu. `complete_retrieval_rate_at_k` là tỷ lệ câu lấy đủ tất cả gold chunk.
Đây là thay đổi tên/ý nghĩa so với artifact schema v1, vốn gọi tỷ lệ đủ mọi
nguồn là recall. Báo cáo mới có `schema_version=2` để phân biệt.

00003 khó vì cần đồng thời nguồn cấp cứu và nguồn khẩn trương. k=1 lấy được
một trong hai: recall riêng câu đó = 0.5, completion = 0. MRR vẫn 1 vì nguồn
đúng đầu tiên vẫn đứng hạng 1; MRR không đo đủ hai nhánh. Router không được
làm recall@3 giảm so với C0.

## Eval, đọc tay và so prompt

```powershell
uv run python eval.py --output artifacts/standard.json --write-review-template
uv run python eval.py --k 1 --output artifacts/k1.json
uv run python eval.py --prompt-variant concise --output artifacts/concise.json
uv run python eval.py --compare artifacts/standard.json artifacts/concise.json --output artifacts/prompt_comparison.json
```

Mỗi lần eval ghi file yêu cầu và một bản lưu có timestamp trong tên để không
mất lịch sử. Report ghi hash code/corpus/QA/prompt, provider/model, số chiều,
tham số generation, phiên bản SDK, top-k, threshold và từng lần cache/API/fallback.
`--compare` in delta retrieval, delta các cờ kiểm tra và những câu đổi nội dung;
nó cũng chỉ ra cấu hình khác ngoài prompt. Synthesis prompt không thay đổi
retrieval: recall có thể giữ nguyên dù cách trả lời thay đổi.

Cột `mechanical_checks_pass` chỉ kiểm quote, citation, thuật ngữ và một số lỗi
số học. Nó **không phải điểm nội dung**; `answer_correct` để null. Không còn
`answerable_full_pass` gây hiểu nhầm. Phép so sánh số sai, thiếu vế so sánh hoặc
liều mượn số từ một nguồn không chứng minh liều đều có regression test.

Sau khi đọc sáu câu với `answer_gt` và context, điền file `.review.json` được
sinh bằng `--write-review-template`: reviewer, ngày giờ, quyết định và lý do từng
câu. Gắn đánh giá vào đúng report, không gọi API lại:

```powershell
uv run python eval.py --review-report artifacts/standard.json --review-file artifacts/standard.review.json
```

Report ID và hash từng câu trả lời ngăn dùng nhầm review cũ. Mọi câu phải có
quyết định cụ thể và ghi chú; template để trống không được chấp nhận.

Kiểm chứng tất cả CLI và ca hội thoại, lưu log riêng:

```powershell
uv run python scripts/verify_assignment.py --offline --output artifacts/check_offline
uv run python scripts/verify_assignment.py --output artifacts/check_live
```

Lệnh thứ hai dùng API đã cấu hình, thử hai prompt và k=1, kiểm các hồi quy.
Hai chế độ dùng cache riêng. Có thể giới hạn tốc độ bằng
`GEMINI_REQUEST_INTERVAL_SECONDS`; lỗi API/quota được ghi là fallback thay vì
báo đạt. File bằng chứng đợt sửa nằm ở `artifacts/verification_20260912/`.

## Một lỗi đã gặp và cách tìm ra

Câu “Còn đái tháo đường cần xét nghiệm gì?” từng bị ghép với câu THA trước đó.
Log `standalone_query` và route cho thấy THA bị giữ lại; hậu kiểm cũ còn bắt câu
trả lời phải chép số của hit đầu (140/90), dù người dùng hỏi xét nghiệm ĐTĐ.
Sửa bằng cách giữ chủ đề rõ ràng của câu mới, bỏ yêu cầu chép mọi số ở hit đầu,
và dùng metadata trong relevance gate để nhận ra tên bệnh đầy đủ khi text viết
“ĐTĐ”. Kiểm thử bắt cả nhầm guideline lẫn việc bị từ chối dù đã tìm đúng nguồn.

## Một câu engine từng trả lời sai nhưng vẫn tự tin

Câu hỏi: **“Giá vàng tăng cao phải làm gì?”**

Bản cũ trả “Theo các đoạn tài liệu được tìm thấy”, rồi nêu “THA khẩn trương:
HA tăng cao NHƯNG chưa có tổn thương cơ quan đích cấp; hạ áp bằng đường uống…”
kèm citation. Nội dung được trích thật nhưng trả lời sai chủ đề, không nói rằng
engine không giải quyết được câu hỏi. Ca này đã tái hiện ở chế độ local trong
`artifacts/audit_20260912/probes.json`, mục `validation`.

Nguyên nhân: validator chỉ cần trùng một từ với corpus; “tăng/cao” đủ để bị
xem là medical. Bản sửa phân loại theo tín hiệu y khoa và chủ đề có nghĩa;
ca này đi thẳng END, không embedding query, không retrieval và không gọi model.
Một lỗi khác đã ghi lại bằng Gemini thật là model từ chối đúng câu hỏi liều
metformin, nhưng engine cũ ép sửa rồi thay bằng context. Bản sửa giữ từ chối,
kể cả model thêm citation vào câu từ chối.

## Giới hạn của kết quả

Golden dataset giữ nguyên. Các rule đọc nội dung corpus/metadata, không đọc
QA gold để sinh câu trả lời. Corpus 8 đoạn và 6 câu chỉ kiểm chứng phạm vi bài
tập; citation hợp lệ và vài kiểm tra số học không chứng minh mọi câu hỏi mới
đều được trả lời đúng. Bộ phân loại dùng tín hiệu ngôn ngữ có giới hạn và
kết quả generation vẫn cần đọc tay. Phần này là lý do giữ ca sai và report
thực tế, không tự quy đổi test pass thành bảo đảm đúng tuyệt đối.
