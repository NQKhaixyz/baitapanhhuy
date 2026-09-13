"""Domain signals and corpus-derived subjects; independent of QA gold."""
from __future__ import annotations
import re
from .embeddings import normalize_text


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalize_text(text)))


MEDICAL_SIGNALS = (
    "huyết áp", "đường huyết", "bệnh", "bệnh nhân", "triệu chứng", "chẩn đoán",
    "xét nghiệm", "điều trị", "chống chỉ định", "tác dụng phụ", "liều", "insulin",
    "egfr", "hba1c", "glucose", "đau", "sốt", "ho", "suy thận", "suy gan",
    "ung thư", "tim mạch", "thai kỳ", "kháng sinh",
)
GREETING = re.compile(r"^(?:xin chào|chào bạn|chào|hello|hi|cảm ơn)(?!\w)[\s,!.:;–-]*", re.I)


def without_greeting(question: str) -> str:
    return GREETING.sub("", question.strip()).strip()


class SubjectIndex:
    def __init__(self, chunks: list[dict]):
        self.aliases: dict[str, tuple[str, str]] = {}
        for chunk in chunks:
            gid = str(chunk.get("guideline_id", ""))
            title = re.split(r"\s+[–—-]\s+", str(chunk.get("guideline_title", "")))[0]
            title = re.sub(r"^HD\s+", "", title, flags=re.I).strip()
            if title:
                self.aliases[normalize_text(title)] = (title, gid)
                abbreviation = "".join(w[0] for w in title.split()).lower()
                if len(abbreviation) >= 2:
                    self.aliases[abbreviation] = (title, gid)
            section = re.split(r"\s+[–—-]\s+", str(chunk.get("section_path", "")))[-1].strip()
            if re.fullmatch(r"[A-Za-z]{4,}", section):
                self.aliases[section.lower()] = (section, gid)

    def subjects(self, question: str) -> list[tuple[str, str]]:
        result = []
        for alias in sorted(self.aliases, key=len, reverse=True):
            subject = self.aliases[alias]
            if contains_phrase(question, alias) and subject not in result:
                result.append(subject)
        return result

    def classify(self, question: str) -> str:
        text = without_greeting(question)
        if question.strip() and not text:
            return "greeting"
        if self.subjects(text) or any(contains_phrase(text, word) for word in MEDICAL_SIGNALS):
            return "medical"
        return "off_topic"
