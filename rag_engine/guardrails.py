"""Các guardrail tổng quát cho answerability và câu trả lời có căn cứ.

Module này không biết tên guideline, bệnh lý hay câu hỏi trong golden dataset.
Nó chỉ dựa trên mức tương đồng, độ phủ thuật ngữ của context và hợp đồng
citation. Vì vậy engine không phải thêm một nhánh đặc biệt cho từng câu bẫy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from .embeddings import normalize_text


TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[A-Za-zÀ-ỹĐđ]+", re.UNICODE)

# Stopword chỉ dùng để đo độ phủ evidence, không dùng để sửa nội dung corpus.
DEFAULT_STOPWORDS = frozenset(
    "a ai à ạ ấy bao bao nhiêu bằng bị bởi cả các cái cần cho có còn của cùng "
    "cũng câu đó được để đến đi điều đâu đúng giữa gì khi không là lại làm mà "
    "mỗi một nào này như những ra sao sẽ theo thì từ và vào với về vậy thế "
    "bệnh nhân người hôm nay này câu nội dung khác"
    .split()
)


def content_terms(text: str, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> set[str]:
    """Trả về các token nội dung ổn định để so query với evidence."""

    ignored = set(stopwords)
    return {
        token
        for token in TOKEN_RE.findall(normalize_text(text))
        if token not in ignored and len(token) > 1
    }


def numeric_terms(text: str) -> set[str]:
    """Lấy các số trong text, chuẩn hóa dấu thập phân đơn giản."""

    return {value.replace(",", ".") for value in re.findall(r"\d+(?:[.,]\d+)?", text)}


@dataclass(frozen=True)
class Answerability:
    """Permission to attempt generation, NOT proof that the answer exists."""

    answerable: bool
    top_score: float
    query_coverage: float
    matched_terms: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "top_score": round(self.top_score, 6),
            "query_coverage": round(self.query_coverage, 6),
            "matched_terms": list(self.matched_terms),
            "reason": self.reason,
        }


def assess_answerability(
    query: str,
    hits: list[dict[str, Any]],
    *,
    threshold: float,
    min_query_coverage: float,
) -> Answerability:
    """Đánh giá relevance bằng điểm cosine và ghi lại lexical coverage.

    Coverage chỉ là telemetry. Dùng nó làm cổng chặn sẽ phạt các câu hỏi dùng
    từ đồng nghĩa (ví dụ ``độ lọc cầu thận`` thay cho ``eGFR``), trong khi
    embedding đã đo độ gần nghĩa. Quyết định nội dung cuối cùng thuộc prompt,
    citation hậu kiểm và manual review.
    """

    if not hits:
        return Answerability(False, 0.0, 0.0, (), "no_hits")

    top_score = float(hits[0].get("score", 0.0))
    if top_score < threshold:
        return Answerability(False, top_score, 0.0, (), "score_below_threshold")

    query_tokens = content_terms(query)
    # Metadata supplies domain/section vocabulary (e.g. full name vs abbreviation).
    # This remains a relevance gate, never evidence for a factual answer.
    context = " ".join(str(hit.get("chunk", {}).get(key, "")) for hit in hits
                       for key in ("guideline_title", "section_path", "text"))
    context_tokens = content_terms(context)
    matched = sorted(query_tokens & context_tokens)
    coverage = len(matched) / len(query_tokens) if query_tokens else 0.0

    if not query_tokens:
        return Answerability(False, top_score, coverage, tuple(matched), "empty_query_terms")
    return Answerability(True, top_score, coverage, tuple(matched), "evidence_ok")


def unsupported_numbers(answer: str, question: str, hits: list[dict[str, Any]]) -> set[str]:
    """Tìm số trong answer không xuất hiện ở question hoặc context.

    Số thứ tự đầu dòng (``1.``, ``2.``...) không bị coi là dosage mới.
    """

    context = " ".join(str(hit.get("chunk", {}).get(key, "")) for hit in hits
                       for key in ("guideline_title", "section_path", "text"))
    allowed = numeric_terms(question) | numeric_terms(context)
    unsupported: set[str] = set()
    answer_without_citations = re.sub(r"\[[^\]]+\]", "", answer)
    for line in answer_without_citations.splitlines():
        stripped = line.lstrip(" -•\t")
        enumeration = re.match(r"^(\d+)[.)]\s", stripped)
        for number in numeric_terms(line):
            if enumeration and number == enumeration.group(1):
                continue
            if number not in allowed:
                unsupported.add(number)
    return unsupported


def refusal_text(answer: str) -> bool:
    """Nhận diện câu từ chối theo các mẫu ngôn ngữ chung."""

    normalized = normalize_text(answer)
    phrases = (
        "không tìm thấy trong tài liệu",
        "không đủ dữ kiện",
        "tài liệu không nêu",
        "không có thông tin",
        "không thể xác định",
    )
    return any(phrase in normalized for phrase in phrases)


REFUSAL = "Không tìm thấy trong tài liệu."


def is_pure_refusal(answer: str) -> bool:
    """Conservative full-string recognition: a disclaimer plus a dose is not refusal."""
    plain = re.sub(r"\[[^\]]+\]", "", answer)
    return normalize_text(plain).strip(" .,!?;`\"'-") in {
        "không tìm thấy trong tài liệu", "không đủ dữ kiện",
        "không có thông tin trong tài liệu", "không thể xác định từ tài liệu",
    }


def is_refusal_clause(text: str) -> bool:
    """A bounded statement about missing information need not invent a citation."""
    plain = normalize_text(re.sub(r"\[[^\]]+\]", "", text)).strip(" -•.?!`")
    if is_pure_refusal(plain):
        return True
    return (plain.startswith(("không tìm thấy trong tài liệu về ", "tài liệu không đề cập ",
                              "tài liệu không nêu ", "không có thông tin trong tài liệu về "))
            and not re.search(r"\d|;|nhưng|tuy nhiên|hãy|nên|có thể", plain))


NUMBER = r"\d+(?:[.,]\d+)?"
COMPARISON = re.compile(
    rf"({NUMBER})\s*(<=|>=|<|>|≤|≥|=|nhỏ hơn|lớn hơn|cao hơn|thấp hơn|"
    rf"dưới|trên|below|above|less than|greater than|bằng)\s*({NUMBER})", re.I
)


def relation_holds(left: str, op: str, right: str) -> bool:
    a, b = float(left.replace(',', '.')), float(right.replace(',', '.'))
    return {"<": a < b, "nhỏ hơn": a < b, "thấp hơn": a < b, "dưới": a < b,
            "below": a < b, "less than": a < b,
            ">": a > b, "lớn hơn": a > b, "cao hơn": a > b, "trên": a > b,
            "above": a > b, "greater than": a > b,
            "<=": a <= b, "≤": a <= b, ">=": a >= b, "≥": a >= b,
            "=": a == b, "bằng": a == b}[op.lower()]


def numeric_violations(question: str, answer: str, hits: list[dict]) -> list[str]:
    """Check explicit arithmetic and measure thresholds; no disease/drug-specific rule."""
    errors = []
    plain = re.sub(r"\[[^\]]+\]", "", answer)
    for left, op, right in COMPARISON.findall(plain):
        if not relation_holds(left, op, right):
            errors.append(f"Phép so sánh sai: {left} {op} {right}.")
    if is_pure_refusal(answer):
        return errors
    # Bind quantities to their cited source, not merely any number in the context.
    by_id = {h['chunk_id']: str(h['chunk'].get('text', '')) for h in hits}
    for clause in re.split(r"\n+|(?<=[.!?;])\s+(?!\[)", answer):
        ids = re.findall(r"\[([^\]]+)\]", clause)
        source = ' '.join(by_id.get(i, '') for i in ids)
        for quantity in re.findall(rf"\b{NUMBER}\s*(?:đơn vị|mg|mcg|µg|IU|U/kg)\b", clause, re.I):
            normalized = re.sub(r"\s+", "", quantity).lower()
            if normalized not in re.sub(r"\s+", "", source).lower():
                errors.append(f"Liều/đơn vị không được nguồn trích dẫn chứng minh: {quantity}.")
    # Only apply a threshold to the same named measure in the question.
    for hit in hits:
        source = str(hit['chunk'].get('text', ''))
        for measure, op, bound in re.findall(rf"\b([A-Za-z][A-Za-z0-9]*)\s*(<=|>=|<|>|≤|≥)\s*({NUMBER})", source):
            value = re.search(rf"\b{re.escape(measure)}\s*(?:=\s*)?({NUMBER})\b", question, re.I)
            if not value or not re.search(r"dùng|sử dụng|chống chỉ định", question, re.I):
                continue
            n = value.group(1)
            if "chống chỉ định" in normalize_text(source) and relation_holds(n, op, bound):
                if not {n.replace(',', '.'), bound.replace(',', '.')}.issubset(numeric_terms(plain)):
                    errors.append(f"Cần áp dụng ngưỡng cho bệnh nhân: {measure} {n} {op} {bound}.")
                if (not re.search(r"không.{0,25}dùng|chống chỉ định", plain, re.I)
                        or re.search(r"(?:có thể|được phép)\s+(?:sử dụng|dùng)", plain, re.I)):
                    errors.append("Kết luận phải phù hợp chống chỉ định trong nguồn.")
    return errors


def required_comparison_sources(question: str, hits: list[dict]) -> set[str]:
    """Identify source branches by their labels, not golden IDs or question IDs."""
    if not re.search(r"khác|so sánh|phân biệt", question, re.I):
        return set()
    query_terms = content_terms(question)
    required = set()
    for hit in hits:
        text = str(hit['chunk'].get('text', ''))
        if ':' not in text:
            continue
        label = text.split(':', 1)[0]
        terms = content_terms(label)
        if len(terms & query_terms) >= 2:
            required.add(hit['chunk_id'])
    return required
