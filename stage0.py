"""Chặng 0 — retriever cosine thuần numpy.

Ví dụ: ``python stage0.py --query "Phân độ tăng huyết áp ra sao?"``
hoặc ``python stage0.py --all`` để chạy sáu câu golden.
"""

from __future__ import annotations

import argparse
import logging

from rag_engine.config import DEFAULT_QA_PATH, TOP_K
from rag_engine.io_utils import read_jsonl
from rag_engine.retrieval import Retriever
from rag_engine.core import gold_chunk_ids
from rag_engine.io_utils import ensure_utf8_output


def main() -> None:
    """Chạy retrieval thủ công hoặc toàn bộ bộ câu hỏi golden.

    CLI in top-k chunk kèm cosine score; với ``--all`` hàm còn tính recall@3
    trên các câu có ``retrieval_gt``. Đây là chặng độc lập để debug retriever
    trước khi thêm generation/router.
    """
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Chặng 0: cosine retrieval")
    parser.add_argument("--query", help="Câu hỏi cần tìm")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ QA golden")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k phải >= 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    retriever = Retriever()
    print(
        f"embedding_provider={type(retriever.provider).__name__} "
        f"model={getattr(retriever.provider, 'model', None)} "
        f"dimension={retriever.vectors.shape[1]}"
    )
    rows = read_jsonl(DEFAULT_QA_PATH) if args.all or not args.query else []
    queries = [(row["id"], row["question"], gold_chunk_ids(row)) for row in rows]
    if args.query:
        queries.append(("manual", args.query, set()))
    if not queries:
        parser.error("Cần --query hoặc --all")
    answerable = 0
    complete_hits = 0
    partial_recall = 0.0
    for item_id, query, gold in queries:
        hits = retriever.search_pairs(query, args.k)
        print(f"\n[{item_id}] {query}")
        for rank, (chunk_id, score) in enumerate(hits, 1):
            print(f"  {rank}. {chunk_id} score={score:.4f}")
        if gold:
            answerable += 1
            retrieved = {chunk_id for chunk_id, _score in hits[: min(args.k, 3)]}
            # Một câu có thể có nhiều chunk vàng (00003). Chỉ tính query hit
            # khi toàn bộ tập chunk cần thiết đã nằm trong top-k.
            complete_hits += int(gold.issubset(retrieved))
            partial_recall += len(gold & retrieved) / len(gold)
    if answerable:
        reported_k = min(args.k, 3)
        print(
            f"\ncomplete_retrieval_rate@{reported_k} (đủ toàn bộ chunk vàng)="
            f"{complete_hits}/{answerable}={complete_hits / answerable:.4f}"
        )
        print(
            f"recall@{reported_k} (tỷ lệ chunk)="
            f"{partial_recall / answerable:.4f}"
        )


if __name__ == "__main__":
    main()
