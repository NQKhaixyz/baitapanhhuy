"""Chặng 1 — synthesis bám context, citation và threshold từ chối."""

from __future__ import annotations

import argparse
import logging

from rag_engine.config import DEFAULT_QA_PATH, RETRIEVAL_THRESHOLD, TOP_K
from rag_engine.evaluation import answer_checks
from rag_engine.io_utils import ensure_utf8_output, read_jsonl
from rag_engine.retrieval import Retriever
from rag_engine.synthesis import GeminiQuotaError, synthesize_answer


def main() -> None:
    """Chạy synthesis grounded cho một query hoặc sáu query golden.

    Mỗi câu được retrieve trước, sau đó qua threshold/guardrail và in answer
    kèm citation. ``answer_checks`` chỉ là kiểm tra tự động hỗ trợ review.
    """
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Chặng 1: grounded answer")
    parser.add_argument("--query")
    parser.add_argument("--all", action="store_true")
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
    if args.query:
        rows.append({"id": "manual", "question": args.query, "meta": {"is_answerable": True}})
    if not rows:
        parser.error("Cần --query hoặc --all")
    try:
        for row in rows:
            question = row["question"]
            hits = retriever.search(question, args.k)
            answerability = retriever.assess_answerability(
                question, hits, RETRIEVAL_THRESHOLD
            )
            answer = synthesize_answer(
                question,
                hits,
                threshold=RETRIEVAL_THRESHOLD,
                answerability=answerability,
            )
            print(f"\n[{row['id']}] {question}\n→ {answer}")
            if row["id"] != "manual":
                print(f"  kiểm tra: {answer_checks(row, answer, hits)}")
    except GeminiQuotaError as exc:
        print(f"\nLỖI QUOTA GEMINI: {exc}")
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        if "GEMINI_API_KEY" in str(exc):
            print("\nTHIẾU GEMINI_API_KEY: hãy đặt biến môi trường trước khi chạy stage1.py.")
            print("PowerShell: $env:GEMINI_API_KEY = \"YOUR_KEY\"")
            raise SystemExit(2) from exc
        raise


if __name__ == "__main__":
    main()
