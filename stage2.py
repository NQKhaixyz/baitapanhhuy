"""Chặng 2 — validate/router và đồ thị sáu node.  
uv run python .\stage2.py --query "tôi bị đau răng " """

from __future__ import annotations

import argparse
import logging

from rag_engine.config import DEFAULT_QA_PATH
from rag_engine.graph import MiniRAGGraph
from rag_engine.io_utils import ensure_utf8_output, read_jsonl
from rag_engine.retrieval import Retriever


def main() -> None:
    """Chạy graph validate/router và in state rút gọn ra terminal."""
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="Chặng 2: validate + route + graph")
    parser.add_argument("--query")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    graph = MiniRAGGraph(Retriever())
    questions = [row["question"] for row in read_jsonl(DEFAULT_QA_PATH)] if args.all else ([args.query] if args.query else ["xin chào", "thời tiết hôm nay thế nào?"])
    for question in questions:
        state = graph.invoke(question)
        print(f"\nQ: {question}\ncategory={state.get('category')} guidelines={state.get('guideline_ids')}\nA: {state['answer']}")
        if state.get("hits"):
            print("context:", [(h["chunk_id"], h["score"]) for h in state["hits"]])


if __name__ == "__main__":
    main()
