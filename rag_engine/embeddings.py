"""Embedding và cache.

Mặc định ưu tiên Gemini khi có ``GEMINI_API_KEY``; nếu chưa có key thì dùng
vector hashing cục bộ để repo vẫn chạy được không cần mạng. Khi muốn dùng OpenAI
thì chọn rõ ``RAG_EMBEDDING_PROVIDER=openai``.
Vector luôn được chuẩn hóa trước khi lưu/tìm, đúng yêu cầu của cosine search.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi trước khi tạo đặc trưng embedding.

    Hàm chỉ đổi Unicode về dạng NFC, chuyển chữ thường và gom khoảng trắng.
    Nội dung gốc không bị sửa trong corpus; bản chuẩn hóa chỉ dùng cho bước
    tính vector để các cách viết tương đương dễ khớp nhau hơn.

    Args:
        text: Chuỗi tiếng Việt cần chuẩn hóa.

    Returns:
        Chuỗi đã chuẩn hóa, không có khoảng trắng thừa.
    """

    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(text.split())


def _features(text: str) -> list[str]:
    """Tách một văn bản thành các đặc trưng cho vector hashing cục bộ.

    Ba nhóm đặc trưng được dùng gồm token đơn, cặp token liên tiếp và
    character n-gram. Không có alias theo domain trong engine; nếu ứng dụng
    cần synonym thì truyền chúng qua provider embedding hoặc cấu hình riêng.

    Args:
        text: Văn bản cần mã hóa.

    Returns:
        Danh sách chuỗi đặc trưng; hàm không gọi mạng và không có side effect.
    """
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    features = [f"w:{token}" for token in tokens]
    features.extend(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    # Character n-gram giúp bắt các biến thể có dấu, viết tắt và số đo.
    padded = f"  {normalized}  "
    features.extend(f"c:{padded[i:i + 3]}" for i in range(max(0, len(padded) - 2)))
    return features


class LocalHashEmbedding:
    """Embedding hashing nhỏ, tái lập, không cần tải model."""

    def __init__(self, dimension: int = 768) -> None:
        """Khởi tạo số chiều của vector hashing.

        Args:
            dimension: Số cột của ma trận vector. Giá trị càng lớn càng giảm
                collision nhưng dùng nhiều bộ nhớ hơn.
        """
        self.dimension = dimension

    def encode(self, texts: Sequence[str], task_type: str | None = None) -> np.ndarray:
        """Mã hóa nhiều văn bản thành vector chuẩn hóa.

        ``task_type`` được nhận để giữ chung interface với Gemini, nhưng local
        embedding không cần phân biệt document/query.

        Args:
            texts: Một hoặc nhiều văn bản cần mã hóa.
            task_type: Loại tác vụ tùy chọn, được bỏ qua ở provider local.

        Returns:
            Ma trận NumPy kích thước ``(len(texts), dimension)`` với mỗi hàng
            có norm bằng 1 (trừ trường hợp văn bản rỗng).
        """
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in _features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimension
                # Signed hashing giảm lệch do collision.
                sign = 1.0 if digest[0] & 1 else -1.0
                matrix[row, index] += sign
        return normalize_rows(matrix)


class OpenAIEmbedding:
    """Adapter tùy chọn; import lazy để offline mode không cần ``openai``."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """Khởi tạo client OpenAI từ ``OPENAI_API_KEY``.

        OpenAI là provider tùy chọn, không được gọi khi repo chạy mặc định với
        Gemini/local. Khởi tạo thất bại sớm nếu thiếu SDK hoặc API key để tránh
        lỗi khó hiểu ở giữa pipeline.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Cài openai để dùng provider OpenAI") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Thiếu OPENAI_API_KEY cho embedding OpenAI")
        self.client = OpenAI()
        self.model = model

    def encode(self, texts: Sequence[str], task_type: str | None = None) -> np.ndarray:
        """Gọi endpoint embedding OpenAI và chuẩn hóa các vector trả về.

        Args:
            texts: Danh sách văn bản gửi trong một request.
            task_type: Tham số tương thích, OpenAI adapter không sử dụng.

        Returns:
            Ma trận vector float32 đã chuẩn hóa theo từng hàng.
        """
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        return normalize_rows(vectors)


class GeminiEmbedding:
    """Adapter Gemini Embeddings API (google-genai, không lưu key trong mã)."""

    def __init__(self, model: str = "gemini-embedding-001") -> None:
        """Khởi tạo Gemini GenAI client bằng biến môi trường.

        API key chỉ đọc từ ``GEMINI_API_KEY``; không ghi key vào source, cache
        hay log. Model mặc định là model embedding Gemini dùng cho retrieval.
        """
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Cài google-genai để dùng embedding Gemini") from exc
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu GEMINI_API_KEY cho embedding Gemini")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        # API đôi khi trả 503/UNAVAILABLE trong vài giây dù key và request đều
        # hợp lệ. Cho phép cấu hình số lần thử để một lần gián đoạn tạm thời
        # không làm hỏng cả lượt đánh giá.
        self.retry_attempts = max(1, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "4")))
        self.retry_base_seconds = max(
            0.1, float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "1.5"))
        )

    def encode(self, texts: Sequence[str], task_type: str | None = None) -> np.ndarray:
        """Sinh embedding Gemini cho document hoặc query.

        Args:
            texts: Các chuỗi cần embedding; corpus có thể gửi theo batch.
            task_type: Ví dụ ``RETRIEVAL_DOCUMENT`` hoặc ``RETRIEVAL_QUERY``.

        Returns:
            Ma trận vector float32 đã chuẩn hóa. Hàm hỗ trợ cả dạng response
            ``embeddings`` mới và ``embedding`` của SDK cũ.
        """
        # Gemini nhận một list contents và trả nhiều Embedding trong một request.
        # Một số phiên bản SDK trả ``embeddings``, phiên bản cũ hơn trả ``embedding``;
        # xử lý cả hai để repo bền hơn khi nâng package.
        config = None
        if task_type:
            from google.genai import types  # type: ignore
            config = types.EmbedContentConfig(task_type=task_type)
        response = None
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.models.embed_content(
                    model=self.model, contents=list(texts), config=config
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                error_text = str(exc)
                transient = (
                    status_code in {408, 429, 500, 502, 503, 504}
                    or any(marker in error_text for marker in (
                        "408", "429", "500", "502", "503", "504",
                        "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED",
                    ))
                )
                if not transient or attempt >= self.retry_attempts - 1:
                    raise RuntimeError(
                        f"Gemini embedding lỗi sau {attempt + 1} lần thử: {error_text}"
                    ) from exc
                retry_after = None
                match = re.search(
                    r"retry in ([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE
                )
                if match:
                    retry_after = float(match.group(1))
                delay = retry_after or self.retry_base_seconds * (2**attempt)
                # Tránh chờ vô hạn nếu server gửi retry-after quá lớn.
                delay = min(delay, 20.0)
                time.sleep(delay)
        if response is None:  # pragma: no cover - vòng lặp chỉ thoát khi đã raise
            raise RuntimeError("Gemini embedding không trả về response")
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            embeddings = [getattr(response, "embedding", response)]
        values = []
        for embedding in embeddings:
            values.append(getattr(embedding, "values", embedding["values"] if isinstance(embedding, dict) else embedding))
        return normalize_rows(np.asarray(values, dtype=np.float32))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Chuẩn hóa từng hàng của ma trận về norm Euclid bằng 1.

    Đây là bước bắt buộc trước cosine search: khi hai vector đã chuẩn hóa,
    tích vô hướng của chúng chính là cosine similarity. Norm rất nhỏ được chặn
    bởi epsilon để không phát sinh chia cho 0.

    Args:
        matrix: Ma trận số thực, trục 0 là mẫu và trục 1 là chiều vector.

    Returns:
        Ma trận cùng kích thước sau chuẩn hóa.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def get_embedding_provider():
    """Chọn provider theo biến môi trường và khả năng chạy offline.

    Thứ tự mặc định là Gemini nếu có ``GEMINI_API_KEY``; nếu không thì dùng
    local hashing. Khi người dùng ép ``RAG_EMBEDDING_PROVIDER=gemini`` hoặc
    ``openai``, lỗi thiếu SDK/key được giữ nguyên để cấu hình sai lộ rõ.

    Returns:
        Một object có phương thức ``encode(texts, task_type=...)``.
    """
    if os.getenv("RAG_OFFLINE") == "1":
        return LocalHashEmbedding()
    provider = os.getenv("RAG_EMBEDDING_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else "local").lower()
    if provider == "gemini":
        try:
            return GeminiEmbedding(os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001"))
        except RuntimeError:
            # Offline fallback có chủ đích; lỗi chi tiết vẫn hiển thị khi user ép
            # provider=gemini nhưng chưa cài SDK/key.
            if os.getenv("RAG_EMBEDDING_PROVIDER") == "gemini":
                raise
            return LocalHashEmbedding()
    if provider == "openai":
        return OpenAIEmbedding(os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"))
    return LocalHashEmbedding()


def load_or_create_embeddings(
    texts: Sequence[str], cache_path: str | Path, provider=None, task_type: str | None = None
) -> np.ndarray:
    """Đọc embedding từ cache hoặc tạo mới đúng một lần cho corpus.

    Vector chính được lưu đúng định dạng ``.npy`` theo yêu cầu bài tập. Một
    file metadata ``.npy.meta.json`` đi kèm giữ fingerprint gồm nội dung,
    provider, model và task type; vì vậy khi đổi từ local sang Gemini hoặc đổi
    model, cache cũ không bị dùng nhầm.

    Args:
        texts: Các văn bản cần vector hóa.
        cache_path: Đường dẫn file ``.npz`` để lưu vector và fingerprint.
        provider: Provider đã khởi tạo; bỏ trống để tự chọn.
        task_type: Task type gửi cho provider (thường là document).

    Returns:
        Ma trận vector tương ứng theo đúng thứ tự ``texts``.
    """

    cache_path = Path(cache_path)
    metadata_path = cache_path.with_name(cache_path.name + ".meta.json")
    provider = provider or get_embedding_provider()
    # Đổi feature schema phải làm cache cũ vô hiệu; nếu không vector local cũ
    # có thể còn alias domain đã bị loại khỏi code.
    provider_key = f"embedding-schema-v2:{type(provider).__name__}:{getattr(provider, 'model', '')}:{getattr(provider, 'dimension', '')}:{task_type or ''}"
    fingerprint = hashlib.sha256(
        (provider_key + "\n" + "\n".join(texts)).encode("utf-8")
    ).hexdigest()
    if cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                cached = np.load(cache_path, allow_pickle=False)
                if cached.ndim == 2 and cached.shape[0] == len(texts) and np.isfinite(cached).all():
                    return normalize_rows(np.asarray(cached, dtype=np.float32))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass
    vectors = provider.encode(texts, task_type=task_type)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return vectors
