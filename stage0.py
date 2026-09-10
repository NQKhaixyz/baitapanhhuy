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
from rag_engine.evaluation import gold_chunk_ids
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    retriever = Retriever()
    rows = read_jsonl(DEFAULT_QA_PATH) if args.all or not args.query else []
    queries = [(row["id"], row["question"], gold_chunk_ids(row)) for row in rows]
    if args.query:
        queries.append(("manual", args.query, set()))
    if not queries:
        parser.error("Cần --query hoặc --all")
    hits_correct = 0
    answerable = 0
    for item_id, query, gold in queries:
        hits = retriever.search(query, args.k)
        print(f"\n[{item_id}] {query}")
        for rank, hit in enumerate(hits, 1):
            print(f"  {rank}. {hit['chunk_id']} score={hit['score']:.4f}")
        if gold:
            answerable += 1
            hits_correct += int(bool(set(hit["chunk_id"] for hit in hits[:3]) & gold))
    if answerable:
        print(f"\nrecall@3={hits_correct}/{answerable}={hits_correct / answerable:.4f}")


if __name__ == "__main__":
    main()
