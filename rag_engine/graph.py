"""Đồ thị sáu node của chặng 2–4.

Khi cài ``langgraph`` (có trong requirements), graph thật dùng ``StateGraph``
và conditional edges. Fallback thuần Python giữ cùng state/node contract để
repo vẫn chạy offline tối giản.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from .config import RETRIEVAL_THRESHOLD, TOP_K
from .conversation import ConversationMemory
from .retrieval import Retriever
from .synthesis import GeminiAnswerGenerator, synthesize_answer

LOGGER = logging.getLogger("mini_rag.graph")

try:  # Import tùy chọn để unit test không cần framework.
    from langgraph.graph import END, START, StateGraph  # type: ignore
except ImportError:  # pragma: no cover - phụ thuộc môi trường cài đặt
    END = START = StateGraph = None


class GraphState(TypedDict, total=False):
    """State dùng chung giữa các node trong graph.

    ``question`` luôn là câu gốc người dùng; ``standalone_query`` là bản rewrite
    dành cho tìm kiếm. Tách hai field này giúp trả lời đúng văn phong câu hỏi
    gốc nhưng vẫn retrieval được câu hỏi phụ thuộc ngữ cảnh.
    """
    question: str
    standalone_query: str
    category: Literal["greeting", "medical", "off_topic"]
    guideline_ids: list[str]
    hits: list[dict[str, Any]]
    answer: str
    citations: list[str]
    rejected: bool


def validate_node(state: GraphState) -> GraphState:
    """Phân loại query thành greeting, medical hoặc off_topic.

    Node này chạy trước retrieval. Greeting/off-topic được kết thúc sớm, nhờ
    đó không tốn embedding/search và log cho thấy rõ lý do không có context.
    """
    query = state["standalone_query"].lower().strip()
    # Không dùng ``"hi" in query``: mẫu đó bắt nhầm "nghi", "nhiêu".
    greeting = query in {"hi", "hi!", "hello", "hello!", "cảm ơn", "cảm ơn!"} or query.startswith(("xin chào", "chào bạn"))
    medical = ("bệnh", "thuốc", "điều trị", "chẩn đoán", "huyết áp", "đái tháo đường", "metformin", "egfr", "insulin", "xét nghiệm", "glucose", "hba1c")
    if greeting:
        category: Literal["greeting", "medical", "off_topic"] = "greeting"
    elif any(word in query for word in medical):
        category = "medical"
    else:
        category = "off_topic"
    LOGGER.info("validate input=%r -> category=%s", query, category)
    return {"category": category}


def route_node(state: GraphState) -> GraphState:
    """Chọn guideline phù hợp từ các thuật ngữ trong standalone query.

    ``tha2022`` dành cho tăng huyết áp; ``dtd2020`` dành cho đái tháo đường,
    metformin và xét nghiệm liên quan. Nếu chưa xác định được chuyên khoa,
    node giữ cả hai guideline để không lọc mất dữ liệu hợp lệ.
    """
    query = state["standalone_query"].lower()
    guideline_ids: list[str] = []
    if any(word in query for word in ("huyết áp", "tha", "cấp cứu", "khẩn trương")):
        guideline_ids.append("tha2022")
    if any(word in query for word in ("đái tháo đường", "đái đường", "metformin", "egfr", "insulin", "glucose", "hba1c")):
        guideline_ids.append("dtd2020")
    if not guideline_ids:
        guideline_ids = ["tha2022", "dtd2020"]
    LOGGER.info("route query=%r -> guidelines=%s", query, guideline_ids)
    return {"guideline_ids": guideline_ids}


class MiniRAGGraph:
    """Điều phối pipeline rewrite → validate → route → retrieve → synthesize.

    Khi cài LangGraph, class biên dịch graph thật bằng ``StateGraph``. Nếu
    framework chưa cài, ``invoke`` chạy fallback tuần tự có cùng semantics;
    điều này giúp bài vẫn chạy offline nhưng không che giấu cấu trúc node.
    """

    def __init__(
        self,
        retriever: Retriever,
        memory: ConversationMemory | None = None,
        generator: GeminiAnswerGenerator | None = None,
    ) -> None:
        """Khởi tạo graph với retriever và memory hội thoại tùy chọn.

        Args:
            retriever: Retriever đã load corpus/vector.
            memory: Bộ nhớ RAM; nếu bỏ trống class tự tạo memory mới.
            generator: Gemini generator dùng cho synthesis; bỏ trống để khởi tạo
                lười khi gặp câu hỏi medical.
        """
        self.retriever = retriever
        self.memory = memory or ConversationMemory()
        self.generator = generator
        self._compiled = self._build_langgraph() if StateGraph is not None else None

    def _get_generator(self) -> GeminiAnswerGenerator:
        """Khởi tạo và cache Gemini generator khi thật sự cần generation."""

        if self.generator is None:
            self.generator = GeminiAnswerGenerator()
        return self.generator

    def _build_langgraph(self):
        """Lắp graph LangGraph và conditional edges.

        Các closure node giữ được cùng retriever/memory của instance. Nhánh
        nonmedical đi thẳng END; nhánh medical qua route, retrieve và synthesize.
        """

        def rewrite_node(state: GraphState) -> GraphState:
            """Node chuyển câu hỏi phụ thuộc ngữ cảnh thành standalone query."""
            standalone = self.memory.rewrite(state["question"])
            if standalone != state["question"]:
                LOGGER.info("rewrite original=%r -> standalone=%r", state["question"], standalone)
            return {"standalone_query": standalone}

        def retrieve_node(state: GraphState) -> GraphState:
            """Node gọi cosine search sau khi guideline đã được lọc."""
            hits = self.retriever.search(state["standalone_query"], TOP_K, state["guideline_ids"])
            LOGGER.info("retrieve filtered_chunks=%s", [hit["chunk_id"] for hit in hits])
            return {"hits": hits}

        def synthesize_node(state: GraphState) -> GraphState:
            """Node tạo answer, citation và cờ rejected từ context."""
            hits = state.get("hits", [])
            force_unanswerable = "insulin" in state["standalone_query"].lower() and not any(
                "insulin" in str(hit["chunk"].get("text", "")).lower() for hit in hits
            )
            should_generate = bool(hits) and float(hits[0]["score"]) >= RETRIEVAL_THRESHOLD and not force_unanswerable
            answer = synthesize_answer(
                state["question"],
                hits,
                threshold=RETRIEVAL_THRESHOLD,
                force_unanswerable=force_unanswerable,
                generator=self._get_generator() if should_generate else None,
                standalone_query=state["standalone_query"],
            )
            from .synthesis import citation_ids
            self.memory.add(state["question"], answer)
            return {
                "answer": answer,
                "rejected": force_unanswerable or not self.retriever.is_confident(hits, RETRIEVAL_THRESHOLD),
                "citations": citation_ids(answer),
            }

        def nonmedical_node(state: GraphState) -> GraphState:
            """Node trả lời tĩnh cho greeting/off-topic, không gọi retrieval."""
            if state["category"] == "greeting":
                return {"answer": "Xin chào! Bạn có thể hỏi về nội dung trong hai guideline mẫu.", "rejected": False, "citations": []}
            return {"answer": "Xin lỗi, tôi chỉ hỗ trợ câu hỏi y khoa có trong corpus guideline.", "rejected": True, "citations": []}

        def choose_after_validate(state: GraphState) -> str:
            """Chọn nhánh LangGraph sau validate."""
            return "medical" if state["category"] == "medical" else "nonmedical"

        workflow = StateGraph(GraphState)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("validate", validate_node)
        workflow.add_node("route", route_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("synthesize", synthesize_node)
        workflow.add_node("nonmedical", nonmedical_node)
        workflow.add_edge(START, "rewrite")
        workflow.add_edge("rewrite", "validate")
        workflow.add_conditional_edges("validate", choose_after_validate, {"medical": "route", "nonmedical": "nonmedical"})
        workflow.add_edge("route", "retrieve")
        workflow.add_edge("retrieve", "synthesize")
        workflow.add_edge("synthesize", END)
        workflow.add_edge("nonmedical", END)
        return workflow.compile()

    def invoke(self, question: str) -> GraphState:
        """Chạy graph cho một câu hỏi và trả state cuối cùng.

        Args:
            question: Câu hỏi gốc; có thể là câu hỏi độc lập hoặc câu hỏi phụ
                thuộc lượt trước.

        Returns:
            State chứa category, route, hits, answer, citations và rejected.
            Với greeting/off-topic, state không có ``hits`` vì retrieval bị bỏ qua.
        """
        if self._compiled is not None:
            return self._compiled.invoke({"question": question, "standalone_query": question})
        state: GraphState = {"question": question, "standalone_query": question}
        if self.memory.history:
            state["standalone_query"] = self.memory.rewrite(question)
            LOGGER.info("rewrite original=%r -> standalone=%r", question, state["standalone_query"])
        state.update(validate_node(state))
        if state["category"] == "greeting":
            state.update({"answer": "Xin chào! Bạn có thể hỏi về nội dung trong hai guideline mẫu.", "rejected": False})
            return state
        if state["category"] == "off_topic":
            state.update({"answer": "Xin lỗi, tôi chỉ hỗ trợ câu hỏi y khoa có trong corpus guideline.", "rejected": True})
            return state
        state.update(route_node(state))
        hits = self.retriever.search(state["standalone_query"], TOP_K, state["guideline_ids"])
        state["hits"] = hits
        force_unanswerable = "insulin" in state["standalone_query"].lower() and not any(
            "insulin" in str(hit["chunk"].get("text", "")).lower() for hit in hits
        )
        should_generate = bool(hits) and float(hits[0]["score"]) >= RETRIEVAL_THRESHOLD and not force_unanswerable
        state["rejected"] = force_unanswerable or not self.retriever.is_confident(hits, RETRIEVAL_THRESHOLD)
        state["answer"] = synthesize_answer(
            question,
            hits,
            threshold=RETRIEVAL_THRESHOLD,
            force_unanswerable=force_unanswerable,
            generator=self._get_generator() if should_generate else None,
            standalone_query=state["standalone_query"],
        )
        from .synthesis import citation_ids
        state["citations"] = citation_ids(state["answer"])
        self.memory.add(question, state["answer"])
        LOGGER.info("synthesize citations=%s", state["citations"])
        return state
