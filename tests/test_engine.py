"""Smoke tests không gọi mạng; dùng local embedding để chạy trong CI."""

from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_PROVIDER", "local")

from rag_engine.config import DEFAULT_CORPUS_PATH
from rag_engine.conversation import ConversationMemory
from rag_engine.graph import MiniRAGGraph
from rag_engine.retrieval import Retriever
from rag_engine.synthesis import synthesize_answer


def test_retriever_returns_known_chunk():
    """Retriever phải xếp chunk phân độ lên đầu với query tương ứng."""
    retriever = Retriever(cache_path=".cache/test_embeddings.npy")
    hits = retriever.search("Phân độ tăng huyết áp gồm mấy độ?", 3)
    assert hits
    assert hits[0]["chunk_id"] == "tha2022_ch2_b2.2"


def test_graph_blocks_off_topic_without_hits():
    """Off-topic phải dừng trước retrieval và không tạo context."""
    graph = MiniRAGGraph(Retriever(cache_path=".cache/test_embeddings.npy"))
    state = graph.invoke("Thời tiết hôm nay thế nào?")
    assert state["category"] == "off_topic"
    assert "thời tiết" not in state.get("answer", "").lower()
    assert "hits" not in state


def test_rewrite_preserves_medical_subject():
    """Rewrite phải giữ chủ đề tăng huyết áp cho câu hỏi nối tiếp."""
    memory = ConversationMemory()
    memory.add("Tăng huyết áp là gì?", "THA là...")
    assert "tăng huyết áp" in memory.rewrite("Thế phân độ ra sao?").lower()


def test_generation_repairs_incomplete_multi_hop_answer():
    """Response Gemini bị cắt phải được yêu cầu viết lại đủ hai nhánh."""

    class SequenceGenerator:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "Cơn khẩn trương có HA"
            return (
                "Cấp cứu có tổn thương cơ quan đích cấp, hạ áp tĩnh mạch "
                "[tha2022_ch7_s7.1]; khẩn trương chưa có tổn thương cơ quan đích cấp, "
                "hạ áp đường uống [tha2022_ch7_s7.2]."
            )

    hits = [
        {"chunk_id": "tha2022_ch7_s7.1", "score": 0.8, "chunk": {"text": ""}},
        {"chunk_id": "tha2022_ch7_s7.2", "score": 0.7, "chunk": {"text": ""}},
    ]
    generator = SequenceGenerator()
    answer = synthesize_answer(
        "Cơn tăng huyết áp cấp cứu khác cơn khẩn trương?",
        hits,
        generator=generator,
    )
    assert generator.calls == 2
    assert "tĩnh mạch" in answer and "đường uống" in answer
