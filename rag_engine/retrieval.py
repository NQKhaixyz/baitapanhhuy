"""Retriever cosine thuần numpy, không dùng FAISS/Chroma/LangChain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_EMBEDDING_CACHE,
    MAX_ROUTE_GUIDELINES,
    MIN_QUERY_COVERAGE,
    RETRIEVAL_THRESHOLD,
    ROUTE_MARGIN,
)
from .embeddings import get_embedding_provider, load_or_create_embeddings, normalize_rows
from .guardrails import content_terms
from .io_utils import read_jsonl
from .core import SubjectIndex

LOGGER = logging.getLogger("mini_rag.retrieve")


class Retriever:
    """Bộ tìm kiếm cosine trên corpus JSONL nhỏ.

    Retriever giữ corpus và ma trận embedding trong RAM. Nó không biết gì về
    prompt hay LLM; nhiệm vụ duy nhất là biến query thành vector, lọc guideline
    nếu được yêu cầu và trả top-k chunk có điểm cao nhất.
    """

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        cache_path: str | Path = DEFAULT_EMBEDDING_CACHE,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Đọc corpus và nạp/tạo vector cho toàn bộ chunk.

        Args:
            corpus_path: File JSONL chứa các chunk đã được gán ``chunk_id``.
            cache_path: File cache embedding; không phải database.
            chunks: Có thể truyền list chunk trực tiếp để test hoặc dùng corpus
                đã được load từ nơi khác.

        Raises:
            ValueError: Nếu corpus rỗng hoặc cache có số vector sai.
        """
        self.corpus_path = Path(corpus_path)
        self.chunks = chunks if chunks is not None else read_jsonl(self.corpus_path)
        if not self.chunks:
            raise ValueError("Corpus không có chunk nào")
        texts = [self._search_text(chunk) for chunk in self.chunks]
        self.provider = get_embedding_provider()
        self.vectors = load_or_create_embeddings(
            texts, cache_path, self.provider, task_type="RETRIEVAL_DOCUMENT"
        )
        if len(self.vectors) != len(self.chunks):
            raise ValueError("Số vector không khớp số chunk")
        self._corpus_terms = set().union(*(content_terms(text) for text in texts))
        self.subjects = SubjectIndex(self.chunks)

    @staticmethod
    def _search_text(chunk: dict[str, Any]) -> str:
        """Ghép metadata có ích vào văn bản được embedding.

        Title và section giúp query ngắn như “phân độ” khớp đúng mục guideline,
        trong khi ``text`` vẫn là nguồn nội dung trả lời cuối cùng.
        """
        return " ".join(
            str(chunk.get(key, ""))
            for key in ("guideline_title", "section_path", "text")
        )

    def _query_vector(self, query: str) -> np.ndarray:
        """Encode query một lần để router và retriever dùng chung."""

        return normalize_rows(
            self.provider.encode([query], task_type="RETRIEVAL_QUERY")
        )[0]

    def _rank_candidates(
        self,
        query_vector: np.ndarray,
        allowed_guideline_ids: Iterable[str] | None = None,
    ) -> list[tuple[int, float]]:
        allowed = set(allowed_guideline_ids) if allowed_guideline_ids is not None else None
        indices = [i for i, chunk in enumerate(self.chunks)
                   if allowed is None or chunk.get("guideline_id") in allowed]
        LOGGER.info("retrieve candidates_before_search=%s", [self.chunks[i]["chunk_id"] for i in indices])
        scores = self.vectors[indices] @ query_vector
        candidates = [
            (index, float(score))
            for index, score in zip(indices, scores)
        ]
        candidates.sort(key=lambda pair: (-pair[1], self.chunks[pair[0]]["chunk_id"]))
        return candidates

    def search(
        self,
        query: str,
        k: int = 3,
        allowed_guideline_ids: Iterable[str] | None = None,
        *,
        query_vector: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm top-k chunk bằng cosine similarity.

        Args:
            query: Câu hỏi hoặc ``standalone_query`` của hội thoại.
            k: Số kết quả tối đa muốn lấy.
            allowed_guideline_ids: Danh sách guideline được router cho phép;
                ``None`` nghĩa là tìm trên toàn corpus.

        Returns:
            Danh sách dict giảm dần theo ``score``. Mỗi phần tử có ``chunk_id``,
            ``score`` và toàn bộ object ``chunk`` để synthesis dùng làm context.
        """

        if k < 1:
            raise ValueError("k phải >= 1")
        query_vector = query_vector if query_vector is not None else self._query_vector(query)
        candidates = self._rank_candidates(query_vector, allowed_guideline_ids)
        hits = [
            {
                "chunk_id": self.chunks[index]["chunk_id"],
                "score": round(score, 6),
                "chunk": self.chunks[index],
            }
            for index, score in candidates[: max(0, k)]
        ]
        LOGGER.info("retrieve query=%r k=%d hits=%s", query, k, [(h["chunk_id"], h["score"]) for h in hits])
        return hits

    def has_lexical_evidence(self, query: str) -> bool:
        """Kiểm tra nhanh query có thuật ngữ chung với corpus hay không.

        Đây là bước validation rẻ, không gọi embedding/API và giúp chặn câu
        ngoài phạm vi trước khi bước retrieve chạy.
        """

        return bool(content_terms(query) & self._corpus_terms)

    def route_guidelines(
        self,
        query: str,
        *,
        max_guidelines: int = MAX_ROUTE_GUIDELINES,
        margin: float = ROUTE_MARGIN,
    ) -> tuple[list[str], dict[str, float], np.ndarray]:
        """Chọn guideline theo điểm embedding và metadata corpus.

        Không biết trước tên guideline hay domain. Guideline đứng đầu được giữ;
        các guideline gần điểm đầu trong ``margin`` cũng được giữ để không làm
        mất context khi câu hỏi liên quan nhiều tài liệu.
        """

        query_vector = self._query_vector(query)
        explicit = list(dict.fromkeys(gid for _, gid in self.subjects.subjects(query)))
        grouped: dict[str, float] = {}
        for index, score in self._rank_candidates(query_vector):
            guideline_id = str(self.chunks[index].get("guideline_id", ""))
            grouped[guideline_id] = max(grouped.get(guideline_id, float("-inf")), score)
        if explicit:
            return explicit, {gid: round(grouped[gid], 6) for gid in explicit}, query_vector
        ranked = sorted(grouped.items(), key=lambda pair: (-pair[1], pair[0]))
        if not ranked:
            return [], {}, query_vector
        top_score = ranked[0][1]
        # Viết vòng lặp rõ ràng để giới hạn số guideline mà không phụ thuộc tên.
        selected = []
        for guideline_id, score in ranked:
            if len(selected) >= max_guidelines:
                break
            if not selected or top_score - score <= margin:
                selected.append((guideline_id, score))
        selected_ids = [guideline_id for guideline_id, _score in selected]
        selected_scores = {guideline_id: round(score, 6) for guideline_id, score in selected}
        return selected_ids, selected_scores, query_vector

    def best_score(self, query: str, allowed_guideline_ids: Iterable[str] | None = None) -> float:
        """Lấy cosine score cao nhất của query sau khi áp dụng filter."""
        hits = self.search(query, 1, allowed_guideline_ids)
        return hits[0]["score"] if hits else 0.0

    def assess_answerability(
        self,
        query: str,
        hits: list[dict[str, Any]],
        threshold: float = RETRIEVAL_THRESHOLD,
        min_query_coverage: float = MIN_QUERY_COVERAGE,
    ):
        """Proxy để caller không cần biết module guardrail nội bộ."""

        from .guardrails import assess_answerability

        return assess_answerability(
            query,
            hits,
            threshold=threshold,
            min_query_coverage=min_query_coverage,
        )

    def search_pairs(
        self,
        query: str,
        k: int = 3,
        allowed_guideline_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Trả đúng contract tối giản ``[(chunk_id, score), ...]``.

        ``search`` giữ thêm toàn bộ chunk để synthesis dùng làm context. Chặng
        retrieval có thể dùng method này khi chỉ cần ID và điểm, không kéo theo
        dữ liệu nội dung trong output.
        """
        return [
            (hit["chunk_id"], float(hit["score"]))
            for hit in self.search(query, k, allowed_guideline_ids)
        ]

    @staticmethod
    def is_confident(hits: list[dict[str, Any]], threshold: float = RETRIEVAL_THRESHOLD) -> bool:
        """Kiểm tra hit đầu có vượt ngưỡng tin cậy để được synthesis hay không.

        Ngưỡng là guardrail chống gọi/sinh câu trả lời từ context không liên
        quan; riêng các câu bẫy còn được synthesis kiểm tra thêm theo thực thể.
        """
        return bool(hits and float(hits[0]["score"]) >= threshold)
