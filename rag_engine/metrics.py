"""Retrieval metrics independent of generation/frameworks."""
from __future__ import annotations
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
    """Macro chunk recall and separate all-source completion; exclude no-gold rows.

    MRR measures only the first relevant result. Recall is averaged per query,
    giving each of the five answerable questions the same weight.
    """

    answerable = [row for row in rows if gold_chunk_ids(row)]
    if not answerable:
        return {
            "answerable_count": 0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "complete_retrieval_rate_at_1": 0.0,
            "complete_retrieval_rate_at_3": 0.0,
            "mrr": 0.0,
        }
    complete1 = complete3 = 0
    partial1 = partial3 = 0.0
    reciprocal_sum = 0.0
    for row in answerable:
        gold = gold_chunk_ids(row)
        predicted = row.get("predicted_chunk_ids", [])
        retrieved1 = set(predicted[:1])
        retrieved3 = set(predicted[:3])
        complete1 += int(gold.issubset(retrieved1))
        complete3 += int(gold.issubset(retrieved3))
        partial1 += len(gold & retrieved1) / len(gold)
        partial3 += len(gold & retrieved3) / len(gold)
        rank = next((index + 1 for index, chunk_id in enumerate(predicted) if chunk_id in gold), None)
        if rank:
            reciprocal_sum += 1.0 / rank
    count = len(answerable)
    return {
        "answerable_count": count,
        "recall_at_1": round(partial1 / count, 4),
        "recall_at_3": round(partial3 / count, 4),
        "complete_retrieval_rate_at_1": round(complete1 / count, 4),
        "complete_retrieval_rate_at_3": round(complete3 / count, 4),
        "mrr": round(reciprocal_sum / count, 4),
    }


