"""Viết lại câu hỏi phụ thuộc ngữ cảnh hội thoại trong RAM."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import MAX_HISTORY
from .embeddings import normalize_text


@dataclass
class ConversationMemory:
    """Bộ nhớ hội thoại tối giản trong RAM.

    Mỗi lượt gồm câu hỏi gốc và câu trả lời trước đó. Chỉ ba lượt gần nhất
    được giữ để rewrite có ngữ cảnh nhưng không làm prompt/history phình to.
    Khi process kết thúc, memory tự mất; đây là chủ ý của bài tập, không phải
    hệ thống multi-user.
    """

    history: list[dict[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        """Thêm một lượt và cắt bỏ các lượt cũ hơn giới hạn.

        Args:
            question: Câu hỏi người dùng vừa gửi, giữ nguyên bản gốc.
            answer: Câu trả lời engine vừa tạo.
        """
        self.history.append({"question": question, "answer": answer})
        del self.history[:-MAX_HISTORY]

    def rewrite(self, question: str) -> str:
        """Đổi câu hỏi phụ thuộc ngữ cảnh thành query độc lập.

        Lượt đầu hoặc câu hỏi đã độc lập được giữ nguyên. Với các mẫu như
        “Thế phân độ ra sao?” và “Còn chống chỉ định?”, hàm lấy chủ đề gần nhất
        trong history để retriever không bị validator hiểu nhầm là off-topic.

        Args:
            question: Câu hỏi mới của người dùng.

        Returns:
            Query độc lập dùng cho validate, route và retrieval.
        """
        if not self.history:
            return question
        normalized = normalize_text(question)
        last = self.history[-1]["question"]
        last_answer = self.history[-1]["answer"]
        # Các mẫu thường gặp trong hội thoại y khoa. Giữ câu hỏi gốc cho phần trả lời.
        if normalized.startswith(("thế ", "vậy ", "còn ", "như vậy")):
            subject = self._subject(last + " " + last_answer)
            return f"{subject}; {question}"
        if normalized in {"còn chống chỉ định?", "chống chỉ định thì sao?", "còn chống chỉ định"}:
            subject = self._subject(last + " " + last_answer)
            return f"{subject}: chống chỉ định là gì?"
        return question

    @staticmethod
    def _subject(text: str) -> str:
        """Đoán chủ đề y khoa gần nhất bằng luật đơn giản, không gọi LLM."""
        normalized = normalize_text(text)
        if "tăng huyết áp" in normalized or "tha" in normalized:
            return "tăng huyết áp"
        if "metformin" in normalized or "đái tháo đường" in normalized:
            return "đái tháo đường và metformin"
        return text.split(".", 1)[0]
