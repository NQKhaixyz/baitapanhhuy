"""Smoke tests không gọi mạng; dùng local embedding để chạy trong CI."""

from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_PROVIDER", "local")

from rag_engine.config import DEFAULT_CORPUS_PATH
from rag_engine.conversation import ConversationMemory
from rag_engine.graph import MiniRAGGraph
from rag_engine.retrieval import Retriever
from rag_engine.synthesis import claims_have_citations, synthesize_answer
from rag_engine.evaluation import answer_checks, retrieval_metrics


def test_retriever_returns_known_chunk():
    """Retriever phải xếp chunk phân độ lên đầu với query tương ứng."""
    retriever = Retriever(cache_path=".cache/test_embeddings.npy")
    hits = retriever.search("Phân độ tăng huyết áp gồm mấy độ?", 3)
    assert hits
    assert hits[0]["chunk_id"] == "tha2022_ch2_b2.2"
    pairs = retriever.search_pairs("Phân độ tăng huyết áp gồm mấy độ?", 1)
    assert pairs[0][0] == "tha2022_ch2_b2.2"
    assert isinstance(pairs[0][1], float)


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
                "Cấp cứu có HA tăng rất cao kèm tổn thương cơ quan đích cấp, hạ áp tĩnh mạch "
                "[tha2022_ch7_s7.1]; khẩn trương chưa có tổn thương cơ quan đích cấp, "
                "hạ áp đường uống [tha2022_ch7_s7.2]."
            )

    hits = [
        {
            "chunk_id": "tha2022_ch7_s7.1",
            "score": 0.8,
            "chunk": {"text": "tăng huyết áp cấp cứu tổn thương cơ quan đích tĩnh mạch"},
        },
        {
            "chunk_id": "tha2022_ch7_s7.2",
            "score": 0.7,
            "chunk": {"text": "tăng huyết áp khẩn trương chưa có tổn thương đường uống"},
        },
    ]
    generator = SequenceGenerator()
    answer = synthesize_answer(
        "Cơn tăng huyết áp cấp cứu khác cơn khẩn trương?",
        hits,
        generator=generator,
    )
    assert generator.calls == 2
    assert "tĩnh mạch" in answer and "đường uống" in answer


def test_multihop_recall_requires_all_gold_chunks():
    """Một câu multi-hop không được pass chỉ vì trúng một nhánh."""
    rows = [
        {
            "id": "item_00003",
            "gold_chunk_ids": ["tha2022_ch7_s7.1", "tha2022_ch7_s7.2"],
            "predicted_chunk_ids": ["tha2022_ch7_s7.1"],
        }
    ]
    metrics = retrieval_metrics(rows)
    assert metrics["complete_retrieval_rate_at_1"] == 0.0
    assert metrics["recall_at_1"] == 0.5


def test_answerable_response_requires_a_citation():
    """Citation rỗng không được lọt qua do tính chất vacuous của all([])."""
    item = {
        "id": "item_00001",
        "meta": {"is_answerable": True},
        "retrieval_gt": [],
    }
    checks = answer_checks(item, "HA tâm thu 140 và tâm trương 90.", [])
    assert checks["citation_ids_valid"] is False
    assert checks["citations_present"] is False


def test_citation_coverage_allows_lead_in_before_cited_bullets():
    """Dòng dẫn nhập kết thúc bằng ':' được cite bởi các bullet ngay sau."""
    answer = "Các ngưỡng chẩn đoán như sau:\n- Glucose ≥7,0 [dtd2020_ch3_s3.1]."
    assert claims_have_citations(answer)
    assert not claims_have_citations("Glucose ≥7,0. HbA1c ≥6,5 [dtd2020_ch3_s3.1].")


def test_conversation_limits_history_and_preserves_topic_switch():
    """Memory chỉ giữ ba lượt và follow-up lấy chủ đề gần nhất."""
    memory = ConversationMemory()
    memory.add("Tăng huyết áp là gì?", "THA là bệnh tăng huyết áp.")
    memory.add("Thế phân độ ra sao?", "Có ba độ THA.")
    memory.add("Metformin dùng khi nào?", "Metformin dùng trong đái tháo đường.")
    memory.add("Câu thứ tư", "Nội dung khác.")
    assert len(memory.history) == 3
    rewritten = memory.rewrite("Còn chống chỉ định?")
    assert "metformin" in rewritten.lower()


def test_threshold_skips_generation_for_low_score():
    """Context dưới threshold phải từ chối trước khi đụng tới generator."""

    class MustNotBeCalled:
        def generate(self, _prompt):  # pragma: no cover - gọi nhầm là test fail
            raise AssertionError("generator không được gọi dưới threshold")

    answer = synthesize_answer(
        "Một câu hỏi ngoài corpus",
        [{"chunk_id": "x", "score": 0.01, "chunk": {"text": "không liên quan"}}],
        threshold=0.2,
        generator=MustNotBeCalled(),
    )
    assert answer == "Không tìm thấy trong tài liệu."


def test_evidence_coverage_skips_generation_for_unseen_question():
    """Score cao nhưng query không có thuật ngữ chung vẫn phải abstain."""

    class MustNotBeCalled:
        def generate(self, _prompt):  # pragma: no cover - gọi nhầm là test fail
            raise AssertionError("query không có evidence mà vẫn gọi generator")

    answer = synthesize_answer(
        "Liều alpha beta cho tình huống chưa có trong tài liệu là bao nhiêu?",
        [{
            "chunk_id": "chunk_a",
            "score": 0.9,
            "chunk": {"text": "Thông tin khác không liên quan đến câu hỏi."},
        }],
        threshold=0.2,
        generator=MustNotBeCalled(),
    )
    assert answer == "Không tìm thấy trong tài liệu."


def test_generation_failure_returns_cited_extractive_fallback():
    """API lỗi không được làm app crash hoặc trả câu không có nguồn."""

    class BrokenGenerator:
        def generate(self, _prompt):
            raise RuntimeError("offline")

    answer = synthesize_answer(
        "Thông tin nào có trong tài liệu?",
        [{
            "chunk_id": "chunk_a",
            "score": 0.9,
            "chunk": {"text": "Một thông tin có số 30 trong tài liệu."},
        }],
        threshold=0.2,
        generator=BrokenGenerator(),
    )
    assert "chunk_a" in answer
    assert claims_have_citations(answer)


def test_empty_guideline_allowlist_does_not_reopen_corpus():
    """Danh sách guideline rỗng phải trả rỗng, không được hiểu thành None."""
    retriever = Retriever(cache_path=".cache/test_embeddings.npy")
    assert retriever.search("tăng huyết áp", 3, allowed_guideline_ids=[]) == []
