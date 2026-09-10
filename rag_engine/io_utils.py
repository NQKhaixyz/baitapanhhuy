"""Các hàm I/O nhỏ, không phụ thuộc framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


def ensure_utf8_output() -> None:
    """Ép stdout/stderr dùng UTF-8 để in tiếng Việt trên Windows.

    Một số PowerShell cũ mở console bằng cp1252. Nếu không đổi encoding, việc
    in câu hỏi có dấu có thể gây ``UnicodeEncodeError`` dù dữ liệu JSONL đúng.
    ``reconfigure`` được gọi có điều kiện để hàm vẫn an toàn trong test runner.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Đọc file JSONL UTF-8 thành list object.

    Args:
        path: Đường dẫn file cần đọc.

    Returns:
        Danh sách dict theo đúng thứ tự dòng trong file.

    Raises:
        ValueError: Nếu một dòng không phải JSON hợp lệ hoặc JSON không phải
            object. Lỗi kèm số dòng để sửa dataset nhanh hơn.
    """

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL lỗi tại {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Dòng {line_no} trong {path} phải là object JSON")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    """Ghi object ra JSON UTF-8 có indent, tự tạo thư mục cha.

    Hàm dùng cho artifact eval để có thể diff hai lần chạy; ``ensure_ascii``
    tắt giúp các ngưỡng và câu trả lời tiếng Việt còn dễ đọc.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def chunks_by_id(chunks: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tạo index ``chunk_id -> chunk`` từ iterable chunk.

    Index này phù hợp khi cần kiểm tra citation hoặc truy xuất metadata mà
    không muốn quét toàn bộ list nhiều lần.
    """
    return {str(chunk["chunk_id"]): chunk for chunk in chunks}
