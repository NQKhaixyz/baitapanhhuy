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
from .core import citation_ids, is_pure_refusal


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
    """Lấy số lâm sàng, bỏ chữ số nằm trong định danh như ``HbA1c``."""

    pattern = r"(?<![A-Za-zÀ-ỹĐđ])\d+(?:[.,]\d+)?"
    return {value.replace(",", ".") for value in re.findall(pattern, text)}


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


def question_intent_violations(question: str, answer: str) -> list[str]:
    """Require a condition or explicit limitation for ``when/indication`` questions."""
    normalized_question = normalize_text(question)
    normalized_answer = normalize_text(re.sub(r"\[[^\]]+\]", "", answer))
    asks_when = bool(re.search(
        r"(?:dùng|sử dụng|chỉ định|điều trị).{0,40}(?:khi nào|lúc nào|bao giờ)|"
        r"(?:khi nào|lúc nào|bao giờ).{0,40}(?:dùng|sử dụng|chỉ định|điều trị)",
        normalized_question,
    ))
    if not asks_when or is_pure_refusal(answer):
        return []
    has_condition_or_limit = bool(re.search(
        r"(?:\bkhi\b|\btrong\b|\bđối với\b|\bchỉ định\b|\bđiều trị\b|"
        r"tài liệu không (?:nêu|đề cập)|không tìm thấy|không đủ dữ kiện)",
        normalized_answer,
    ))
    if has_condition_or_limit:
        return []
    return [
        "Câu hỏi hỏi thời điểm/chỉ định nhưng câu trả lời chưa nêu điều kiện "
        "hoặc giới hạn thông tin của tài liệu."
    ]


REFUSAL = "Không tìm thấy trong tài liệu."


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
    # Keep semicolon-separated bullet clauses together because a citation at
    # the end of the bullet supports the whole line (same contract as core).
    for clause in re.split(r"\n+|(?<=[.!?])\s+(?!\[)", answer):
        ids = citation_ids(clause)
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
            if "chống chỉ định" not in normalize_text(source):
                continue
            required_numbers = {n.replace(',', '.'), bound.replace(',', '.')}
            if not required_numbers.issubset(numeric_terms(plain)):
                errors.append(f"Cần áp dụng ngưỡng cho bệnh nhân: {measure} {n} so với {bound}.")
            if relation_holds(n, op, bound):
                if (not re.search(r"không.{0,25}dùng|chống chỉ định", plain, re.I)
                        or re.search(r"(?:có thể|được phép)\s+(?:sử dụng|dùng)", plain, re.I)):
                    errors.append("Kết luận phải phù hợp chống chỉ định trong nguồn.")
            else:
                normalized_plain = normalize_text(plain)
                explicit_outside = re.search(
                    r"không.{0,35}(?:thuộc|nằm|bị).{0,35}chống chỉ định|"
                    r"không chống chỉ định",
                    normalized_plain,
                )
                permission = re.search(
                    r"(?:có thể|được)\s+(?:sử dụng|dùng)|(?:sử dụng|dùng)\s+được",
                    normalized_plain,
                )
                uncertain = re.search(
                    r"(?:không nêu rõ|không thể xác định|chưa thể xác định)"
                    r".{0,50}(?:sử dụng|dùng)",
                    normalized_plain,
                )
                if not explicit_outside and (not permission or uncertain):
                    errors.append(
                        "Cần kết luận rõ giá trị không thuộc ngưỡng chống chỉ định theo tiêu chí trong nguồn."
                    )
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
