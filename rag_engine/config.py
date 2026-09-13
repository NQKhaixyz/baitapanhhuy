"""Cấu hình đường dẫn và tham số dùng chung cho toàn bộ các chặng."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - dependency cÃ³ trong requirements
    load_dotenv = None
if load_dotenv is not None:
    # Nạp .env từ thư mục gốc; biến PowerShell đã có sẵn được giữ nguyên.
    load_dotenv(ROOT_DIR / ".env", override=False)
DATASET_DIR = ROOT_DIR / "golden-dataset"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
CACHE_DIR = ROOT_DIR / ".cache"
DEFAULT_CORPUS_PATH = DATASET_DIR / "chunks.corpus.jsonl"
DEFAULT_QA_PATH = DATASET_DIR / "qa.sample.jsonl"
DEFAULT_EMBEDDING_CACHE = Path(os.getenv("RAG_EMBEDDING_CACHE", str(CACHE_DIR / "corpus_embeddings.npy")))
DEFAULT_EVAL_OUTPUT = ARTIFACT_DIR / "eval_results.json"

TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RETRIEVAL_THRESHOLD = float(os.getenv("RAG_RETRIEVAL_THRESHOLD", "0.18"))
MIN_QUERY_COVERAGE = float(os.getenv("RAG_MIN_QUERY_COVERAGE", "0.20"))
ROUTE_MARGIN = float(os.getenv("RAG_ROUTE_MARGIN", "0.08"))
MAX_ROUTE_GUIDELINES = max(1, int(os.getenv("RAG_MAX_ROUTE_GUIDELINES", "2")))
MAX_HISTORY = 3


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


# Các rule ngôn ngữ nằm trong config thay vì rải trong node/domain code.
GREETING_EXACT = _csv_env(
    "RAG_GREETING_EXACT",
    ("hi", "hi!", "hello", "hello!", "cảm ơn", "cảm ơn!"),
)
GREETING_PREFIXES = _csv_env("RAG_GREETING_PREFIXES", ("xin chào", "chào bạn"))
CONTEXTUAL_PREFIXES = _csv_env(
    "RAG_CONTEXTUAL_PREFIXES",
    ("thế ", "vậy ", "còn ", "như vậy", "và "),
)
CONTEXTUAL_EXACT = _csv_env(
    "RAG_CONTEXTUAL_EXACT",
    ("này", "đó", "chống chỉ định?", "chống chỉ định"),
)
GREETING_RESPONSE = os.getenv(
    "RAG_GREETING_RESPONSE",
    "Xin chào! Bạn có thể hỏi về nội dung trong corpus guideline.",
)
OFF_TOPIC_RESPONSE = os.getenv(
    "RAG_OFF_TOPIC_RESPONSE",
    "Xin lỗi, tôi chỉ hỗ trợ câu hỏi có bằng chứng trong corpus guideline.",
)
