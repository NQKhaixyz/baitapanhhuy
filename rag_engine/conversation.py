"""Three-turn RAM memory with a resolved subject per turn."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from .config import CONTEXTUAL_EXACT, CONTEXTUAL_PREFIXES, DEFAULT_CORPUS_PATH, MAX_HISTORY
from .embeddings import normalize_text
from .io_utils import read_jsonl
from .language import SubjectIndex


# Follow-up ngắn thường bỏ tên bệnh/thuốc ("Liều dùng bao nhiêu?", "Phân
# độ ra sao?"). Chỉ nối ngữ cảnh khi mọi token đều là từ hỏi chung; nhờ vậy
# một chủ đề mới như "liều amlodipine..." vẫn được giữ nguyên để validator và
# router xử lý độc lập.
GENERIC_FOLLOWUP_WORDS = frozenset(
    "liều dùng sử dụng bao nhiêu khi nào chống chỉ định tác dụng phụ phân độ "
    "ngưỡng xét nghiệm điều trị triệu chứng khởi đầu là có được không ra sao "
    "thế nào như thế nào này đó nó thì và với egfr glucose hba1c mg ml phút "
    "mmhg iu u kg bệnh nhân người cần cho thuốc".split()
)


@dataclass
class ConversationMemory:
    history: list[dict[str, str]] = field(default_factory=list)
    subjects: SubjectIndex = field(default_factory=lambda: SubjectIndex(read_jsonl(DEFAULT_CORPUS_PATH)))

    def add(self, question: str, answer: str, standalone_query: str | None = None) -> None:
        resolved = standalone_query or self.rewrite(question)
        names = [name for name, _ in self.subjects.subjects(resolved)]
        self.history.append({"question": question, "answer": answer,
                             "standalone_query": resolved, "subject": ", ".join(names)})
        del self.history[:-MAX_HISTORY]

    def rewrite(self, question: str) -> str:
        if not self.history:
            return question
        if self.subjects.subjects(question):
            # A newly named subject is standalone even after a discourse connector.
            return re.sub(r"^(?:còn|vậy|thế|và)\s+", "", question.strip(), flags=re.I)
        if not self._is_context_dependent(normalize_text(question)):
            return question
        for turn in reversed(self.history):
            subject = turn.get("subject")
            if not subject:
                subject = ", ".join(name for name, _ in self.subjects.subjects(
                    turn.get("standalone_query", turn["question"])))
            if subject:
                return f"Về {subject}: {question}"
        return question

    @staticmethod
    def _is_context_dependent(normalized: str) -> bool:
        if (normalized.startswith(CONTEXTUAL_PREFIXES) or normalized in CONTEXTUAL_EXACT
                or any(word in normalized.split() for word in ("này", "đó", "nó"))):
            return True
        # A short medical follow-up often omits the entity. Reject any token
        # outside this vocabulary so a newly named drug/topic is not attached
        # to the previous turn by accident.
        tokens = re.findall(r"[\wÀ-ỹĐđ]+", normalized)
        return (len(tokens) <= 8 and bool(tokens)
                and any(token in GENERIC_FOLLOWUP_WORDS for token in tokens)
                and all(token in GENERIC_FOLLOWUP_WORDS or token.isdigit() for token in tokens))
