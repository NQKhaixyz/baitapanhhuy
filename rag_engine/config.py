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
DEFAULT_EMBEDDING_CACHE = CACHE_DIR / "corpus_embeddings.npy"
DEFAULT_EVAL_OUTPUT = ARTIFACT_DIR / "eval_results.json"

TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RETRIEVAL_THRESHOLD = float(os.getenv("RAG_RETRIEVAL_THRESHOLD", "0.18"))
MAX_HISTORY = 3
