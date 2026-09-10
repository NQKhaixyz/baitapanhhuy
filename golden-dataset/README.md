# Golden Dataset — AI4Health (RAG y tế)

Bộ dữ liệu chuẩn do bác sĩ tạo, dùng để đánh giá pipeline RAG ở 2 tầng: **retrieval** (lấy đúng đoạn guideline) và **generation** (trả lời đúng).

📄 **Tài liệu thiết kế đầy đủ:** https://claude.ai/code/artifact/2cf1fe8b-32a0-43ab-a28b-0c8eb02404a4

## File trong thư mục

| File | Nội dung |
|------|----------|
| `qa.sample.jsonl` | 6 câu hỏi mẫu (1 dòng = 1 câu) theo schema v1.0 |
| `chunks.corpus.jsonl` | 8 chunk guideline mẫu — `chunk_ids` trong qa trỏ về đây |
| `README.md` | File này |

## Nguyên tắc lưu trữ

- **Thu thập** → database quan hệ (SQLite/Postgres): ràng buộc 5 câu/loại, review, sửa/xóa mềm.
- **Đánh giá** → xuất ra **JSONL** bất biến, có `dataset_version`. Không dùng CSV (dữ liệu lồng nhau sẽ vỡ).

## Điểm cốt lõi: neo về `chunk_id`

Guideline được chunk hóa & gán ID trước (`chunks.corpus.jsonl`). Bác sĩ **chọn đúng đoạn** khi trích dẫn → lưu `chunk_ids`. Nhờ đó eval retrieval (recall@k, MRR) chạy tự động, khớp trực tiếp với chunk mà retriever trả về.

## Các trường chính (qa.jsonl)

| Trường | Dùng cho |
|--------|----------|
| `retrieval_gt[].chunk_ids` | Recall@k, Precision@k, MRR của retriever |
| `retrieval_gt[].points[]` | Point-coverage: context/đáp án có đủ ý không |
| `points[].verbatim_quote` | Khớp text dự phòng + chấm faithfulness |
| `answer_gt` | Chấm đúng/sai đáp án |
| `meta.is_answerable` | Câu bẫy — chatbot phải biết từ chối, không bịa |
| `type_id`, `specialty` | Slice điểm theo loại / chuyên khoa |
| `dataset_version` | Cố định snapshot để so điểm qua các lần chạy |
| `review.status` | Chỉ eval câu `approved` |

## Đọc thử nhanh (Python)

```python
import json
qa     = [json.loads(l) for l in open("qa.sample.jsonl", encoding="utf-8")]
chunks = {c["chunk_id"]: c for c in
          (json.loads(l) for l in open("chunks.corpus.jsonl", encoding="utf-8"))}

for item in qa:
    gt_chunks = [cid for c in item["retrieval_gt"] for cid in c["chunk_ids"]]
    print(item["id"], "→ cần lấy:", gt_chunks)
```

> Nội dung ngưỡng/guideline trong file là **ví dụ minh họa**, thay bằng dữ liệu thật khi triển khai. Schema là phần cần giữ.
