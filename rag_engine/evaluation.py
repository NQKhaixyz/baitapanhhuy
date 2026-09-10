"""Metric retrieval và các kiểm tra an toàn cho bộ golden."""

from __future__ import annotations

import re
from typing import Any


def gold_chunk_ids(item: dict[str, Any]) -> set[str]:
    """Lấy tập chunk vàng từ một dòng QA hoặc một dòng kết quả eval.

    Hàm hỗ trợ cả schema gốc ``retrieval_gt`` và schema trung gian có sẵn
    ``gold_chunk_ids`` để metric có thể chạy sau khi pipeline đã xử lý row.
    """
    if "gold_chunk_ids" in item:
        return set(item["gold_chunk_ids"])
    return {
        chunk_id
        for group in item.get("retrieval_gt", [])
        for chunk_id in group.get("chunk_ids", [])
    }


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Tính recall@1, recall@3 và MRR theo chunk_id.

    Câu không có gold retrieval (ví dụ câu hỏi liều insulin) không bị tính là
    miss retrieval; chúng được đánh giá riêng bởi guardrail refusal.

    Args:
        rows: Danh sách QA gốc hoặc kết quả có ``predicted_chunk_ids``.

    Returns:
        Dict metric dạng số, kèm ``answerable_count`` làm mẫu số rõ ràng.
    """

    answerable = [row for row in rows if gold_chunk_ids(row)]
    if not answerable:
        return {"answerable_count": 0, "recall_at_1": 0.0, "recall_at_3": 0.0, "mrr": 0.0}
    hit1 = hit3 = 0
    reciprocal_sum = 0.0
    for row in answerable:
        gold = gold_chunk_ids(row)
        predicted = row.get("predicted_chunk_ids", [])
        if set(predicted[:1]) & gold:
            hit1 += 1
        if set(predicted[:3]) & gold:
            hit3 += 1
        rank = next((index + 1 for index, chunk_id in enumerate(predicted) if chunk_id in gold), None)
        if rank:
            reciprocal_sum += 1.0 / rank
    count = len(answerable)
    return {
        "answerable_count": count,
        "recall_at_1": round(hit1 / count, 4),
        "recall_at_3": round(hit3 / count, 4),
        "mrr": round(reciprocal_sum / count, 4),
    }


def answer_checks(item: dict[str, Any], answer: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Kiểm tra các điều kiện generation có thể tự động hóa.

    Hàm kiểm tra refusal cho câu không answerable, các thuật ngữ bắt buộc cho
    5 câu còn lại, quote trong context và citation thuộc top-k. Đây không phải
    đánh giá lâm sàng thay bác sĩ; ``answer_gt_reference`` chỉ được lưu để
    người review đối chiếu.

    Args:
        item: Một dòng trong golden QA.
        answer: Câu trả lời engine sinh ra.
        hits: Context retriever đã đưa cho synthesis.

    Returns:
        Dict cờ kiểm tra và thông tin tham chiếu cho artifact eval.
    """

    gold = item.get("answer_gt", "")
    is_answerable = bool(item.get("meta", {}).get("is_answerable", True))
    context = " ".join(str(hit["chunk"].get("text", "")) for hit in hits)
    cited = re.findall(r"\[([^\]]+)\]", answer)
    valid_ids = {hit["chunk_id"] for hit in hits}
    if not is_answerable:
        refusal = any(term in answer.lower() for term in ("không đủ dữ kiện", "không nêu", "không nên tự suy ra"))
        return {"answerable": False, "correct_refusal": refusal, "citation_ids_valid": all(x in valid_ids for x in cited)}

    # Kiểm tra điểm/ý quan trọng thay vì so chuỗi cứng toàn câu.
    expected_terms = {
        "item_00001": ("140", "90"),
        "item_00002": ("độ 1", "độ 2", "độ 3"),
        "item_00003": ("tổn thương cơ quan đích", "tĩnh mạch", "đường uống"),
        "item_00004": ("glucose", "hba1c", "ogtt", "7,0", "6,5", "11,1"),
        "item_00005": ("không", "20 < 30", "<30"),
    }.get(item.get("id"), ())
    lower_answer = answer.lower()
    if item.get("id") == "item_00005":
        # Cho phép khoảng trắng tùy cách model viết toán tử, nhưng vẫn bắt
        # buộc phải có cả phép tính 20 < 30 và ngưỡng chống chỉ định <30.
        terms_ok = (
            "không" in lower_answer
            and re.search(r"20\s*<\s*30", lower_answer) is not None
            and re.search(r"<\s*30", lower_answer) is not None
        )
    else:
        terms_ok = all(term.lower() in lower_answer for term in expected_terms)
    quotes_ok = all(point.get("verbatim_quote", "") in context for group in item.get("retrieval_gt", []) for point in group.get("points", []))
    return {
        "answerable": True,
        "expected_terms_ok": terms_ok,
        "verbatim_quotes_in_context": quotes_ok,
        "citation_ids_valid": all(x in valid_ids for x in cited),
        "answer_gt_reference": gold,
    }
