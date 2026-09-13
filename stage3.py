"""Chặng 3 — hội thoại nhiều lượt, rewrite trước validate.

Ví dụ: ``uv run python .\\stage3.py``
"""

from __future__ import annotations

import argparse
import logging

from rag_engine.conversation import ConversationMemory
from rag_engine.graph import MiniRAGGraph
from rag_engine.io_utils import ensure_utf8_output
from rag_engine.retrieval import Retriever


def main() -> None:
    """Chạy chuỗi hội thoại mẫu để quan sát câu gốc và standalone query."""
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Chặng 3: conversational rewrite")
    parser.add_argument("--query", action="append", help="Có thể truyền nhiều --query")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    questions = args.query or [
        "Tăng huyết áp là gì?",
        "Thế phân độ ra sao?",
        "Metformin dùng khi nào?",
        "Còn chống chỉ định?",
    ]
    graph = MiniRAGGraph(Retriever(), ConversationMemory())
    for question in questions:
        before = graph.memory.history[-1]["question"] if graph.memory.history else "(trống)"
        state = graph.invoke(question)
        print(f"\nCâu gốc: {question}\nCâu trước: {before}\nCâu tìm kiếm độc lập: {state['standalone_query']}\n→ {state['answer']}")


if __name__ == "__main__":
    main()
