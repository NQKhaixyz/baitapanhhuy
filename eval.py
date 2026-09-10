"""Bộ chấm retrieval + các guardrail generation.

Lệnh chính: ``python eval.py``. Kết quả ghi vào ``artifacts/eval_results.json``.
Golden dataset chỉ được đọc, không bị sửa.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from rag_engine.config import DEFAULT_EVAL_OUTPUT, DEFAULT_QA_PATH, RETRIEVAL_THRESHOLD, TOP_K
from rag_engine.evaluation import answer_checks, gold_chunk_ids, retrieval_metrics
from rag_engine.graph import MiniRAGGraph
from rag_engine.io_utils import ensure_utf8_output, read_jsonl, write_json
from rag_engine.retrieval import Retriever
from rag_engine.synthesis import GeminiQuotaError


def run_eval(output_path=DEFAULT_EVAL_OUTPUT) -> dict:
    """Chạy toàn bộ golden dataset và ghi report JSON.

    Args:
        output_path: File artifact đầu ra; thư mục cha sẽ được tạo tự động.

    Returns:
        Report gồm timestamp, retrieval metrics, answer checks và kết quả từng
        item. Dataset gốc chỉ được đọc, không bị sửa.
    """
    rows = read_jsonl(DEFAULT_QA_PATH)
    retriever = Retriever()
    graph = MiniRAGGraph(retriever)
    results = []
    for row in rows:
        # 00006 vẫn chạy qua toàn graph để đo refusal, nhưng không có gold retrieval.
        state = graph.invoke(row["question"])
        hits = state.get("hits", [])
        predicted = [hit["chunk_id"] for hit in hits]
        result = {
            "id": row["id"],
            "question": row["question"],
            "gold_chunk_ids": sorted(gold_chunk_ids(row)),
            "predicted_chunk_ids": predicted,
            "scores": {hit["chunk_id"]: hit["score"] for hit in hits},
            "answer": state["answer"],
            "checks": answer_checks(row, state["answer"], hits),
        }
        results.append(result)
    metrics = retrieval_metrics(results)
    answer_summary = {
        "correct_refusal": sum(bool(item["checks"].get("correct_refusal")) for item in results if not item["checks"].get("answerable", True)),
        "answerable_term_pass": sum(bool(item["checks"].get("expected_terms_ok")) for item in results if item["checks"].get("answerable")),
        "citation_valid": sum(bool(item["checks"].get("citation_ids_valid")) for item in results),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DEFAULT_QA_PATH),
        "retrieval_threshold": RETRIEVAL_THRESHOLD,
        "top_k": TOP_K,
        "metrics": metrics,
        "answer_summary": answer_summary,
        "items": results,
    }
    write_json(output_path, report)
    return report


def main() -> None:
    """Phân tích tham số CLI, chạy eval và in metric tóm tắt."""
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Eval Mini RAG Engine")
    parser.add_argument("--output", default=str(DEFAULT_EVAL_OUTPUT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    try:
        report = run_eval(args.output)
    except GeminiQuotaError as exc:
        print(f"LỖI QUOTA GEMINI: {exc}")
        print("Hãy chờ quota reset, dùng gemini-3.5-flash-lite hoặc nâng billing.")
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        if "GEMINI_API_KEY" in str(exc):
            print("THIẾU GEMINI_API_KEY: hãy đặt biến môi trường trước khi chạy eval.py.")
            print("PowerShell: $env:GEMINI_API_KEY = \"YOUR_KEY\"")
            raise SystemExit(2) from exc
        raise
    print("Retrieval metrics:")
    for key, value in report["metrics"].items():
        print(f"  {key}: {value}")
    print("Answer checks:", report["answer_summary"])
    print(f"Đã ghi: {args.output}")


if __name__ == "__main__":
    main()
