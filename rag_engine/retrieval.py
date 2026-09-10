"""Retriever cosine thuần numpy, không dùng FAISS/Chroma/LangChain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import DEFAULT_CORPUS_PATH, DEFAULT_EMBEDDING_CACHE, RETRIEVAL_THRESHOLD
from .embeddings import get_embedding_provider, load_or_create_embeddings, normalize_rows
from .io_utils import read_jsonl

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

    def search(
        self,
        query: str,
        k: int = 3,
        allowed_guideline_ids: Iterable[str] | None = None,
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

        query_vector = normalize_rows(
            self.provider.encode([query], task_type="RETRIEVAL_QUERY")
        )[0]
        scores = self.vectors @ query_vector
        allowed = set(allowed_guideline_ids) if allowed_guideline_ids else None
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if allowed is None or self.chunks[index].get("guideline_id") in allowed
        ]
        candidates.sort(key=lambda pair: (-pair[1], self.chunks[pair[0]]["chunk_id"]))
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

    def best_score(self, query: str, allowed_guideline_ids: Iterable[str] | None = None) -> float:
        """Lấy cosine score cao nhất của query sau khi áp dụng filter."""
        hits = self.search(query, 1, allowed_guideline_ids)
        return hits[0]["score"] if hits else 0.0

    @staticmethod
    def is_confident(hits: list[dict[str, Any]], threshold: float = RETRIEVAL_THRESHOLD) -> bool:
        """Kiểm tra hit đầu có vượt ngưỡng tin cậy để được synthesis hay không.

        Ngưỡng là guardrail chống gọi/sinh câu trả lời từ context không liên
        quan; riêng các câu bẫy còn được synthesis kiểm tra thêm theo thực thể.
        """
        return bool(hits and float(hits[0]["score"]) >= threshold)
