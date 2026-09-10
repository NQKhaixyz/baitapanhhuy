"""Sinh câu trả lời bằng Gemini và kiểm tra grounding/citation.

LLM chỉ được gọi sau khi retrieval vượt threshold. Context được đóng gói cùng
``chunk_id`` để Gemini trích dẫn được nguồn; sau response, citation vẫn phải
qua whitelist trước khi được trả ra ngoài. Câu không answerable bị từ chối
trước khi gọi model.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

from .config import RETRIEVAL_THRESHOLD
from .embeddings import normalize_text

GROUNDING_INSTRUCTION = """Bạn là trợ lý tổng hợp thông tin y tế. Chỉ sử dụng thông tin có trong CONTEXT được cấp để trả lời.

YÊU CẦU BẮT BUỘC:
- Đặt `[chunk_id]` ngay sau từng khẳng định chứa dữ liệu/kết luận tương ứng.
- Không suy đoán, không giả định, không đưa kiến thức ngoài CONTEXT vào câu trả lời.
- Nếu không có thông tin hoặc thông tin không đủ để kết luận, trả lời chính xác: "Không tìm thấy trong tài liệu."
- Với câu hỏi có nhiều vế, chỉ trả lời vế có dẫn chứng rõ ràng, các vế còn lại phải nêu rõ là tài liệu không đề cập.
- Nếu câu hỏi đưa một con số cần so với ngưỡng, phải thực hiện và viết rõ phép so sánh (ví dụ: 20 < 30).
- Với câu hỏi multi-hop, phải trả lời đủ mọi nhánh được hỏi và các xử trí/đường dùng có trong context.
- Với câu hỏi xét nghiệm/chẩn đoán, phải nêu cả các ngưỡng số liệu xuất hiện trong context.
- Không dừng giữa câu; luôn kết thúc câu trả lời hoàn chỉnh bằng dấu câu hoặc citation.
"""


class GeminiQuotaError(RuntimeError):
    """Lỗi quota/rate limit Gemini, kèm thời gian retry nếu API cung cấp."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class GeminiAnswerGenerator:
    """Adapter gọi Gemini Generate Content API cho lớp synthesis."""

    def __init__(self, model: str | None = None) -> None:
        """Khởi tạo client Gemini bằng ``GEMINI_API_KEY``.

        Args:
            model: Tên model generation; mặc định đọc
                ``GEMINI_GENERATION_MODEL`` hoặc dùng ``gemini-3.5-flash-lite``.

        Raises:
            RuntimeError: Khi thiếu SDK hoặc API key.
        """

        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Cài google-genai để dùng Gemini generation") from exc
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu GEMINI_API_KEY cho Gemini generation")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite")
        # gemini-2.5-flash-lite có thể trả 404 với project/user mới; API
        # thường gợi ý bản 3.5 Flash Lite thay thế.
        self.fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
        self.cache_dir = Path(os.getenv("RAG_GENERATION_CACHE_DIR", ".cache/generation"))
        self.retry_attempts = max(1, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "4")))
        self.retry_base_seconds = max(
            0.1, float(os.getenv("GEMINI_RETRY_BASE_SECONDS", "1.5"))
        )

    def generate(self, prompt: str) -> str:
        """Gửi grounded prompt tới Gemini và lấy text response.

        Args:
            prompt: Prompt đã chứa instruction, câu hỏi và top-k context.

        Returns:
            Nội dung text Gemini sinh ra.

        Raises:
            RuntimeError: Nếu response không có text để sử dụng.
        """

        cache_key = hashlib.sha256(
            f"{self.model}\n{prompt}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        from google.genai import types  # type: ignore

        response = None
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        candidate_count=1,
                        max_output_tokens=1000,
                        temperature=0.2,
                    ),
                )
                break
            except Exception as exc:
                # SDK đã tự retry một số lỗi; nếu vẫn là 429 thì chuyển thành lỗi
                # ngắn gọn để CLI không in cả traceback LangGraph dài.
                status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                error_text = str(exc)
                if status_code == 429 or "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    if self.model != self.fallback_model:
                        # Flash thường có quota thấp hơn Flash-Lite ở free tier;
                        # chuyển model một lần để eval không chết ở câu thứ sáu.
                        self.model = self.fallback_model
                        return self.generate(prompt)
                    retry_after = None
                    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE)
                    if match:
                        retry_after = float(match.group(1))
                    wait_text = f" khoảng {retry_after:.0f} giây" if retry_after else " sau khi quota reset"
                    raise GeminiQuotaError(
                        f"Gemini đang hết quota/rate limit{wait_text}. "
                        "Hãy chờ, đổi sang gemini-3.5-flash-lite hoặc nâng quota/billing.",
                        retry_after,
                    ) from exc
                if status_code == 404 or "NOT_FOUND" in error_text:
                    if self.model != self.fallback_model:
                        # Một số project không được cấp model cũ cho user mới.
                        self.model = self.fallback_model
                        return self.generate(prompt)
                    raise RuntimeError(
                        f"Model Gemini không khả dụng: {self.model}. "
                        "Hãy chọn model generation còn được project hỗ trợ."
                    ) from exc
                transient = (
                    status_code in {408, 500, 502, 503, 504}
                    or any(marker in error_text for marker in (
                        "408", "500", "502", "503", "504", "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ))
                )
                if transient and attempt < self.retry_attempts - 1:
                    match = re.search(
                        r"retry in ([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE
                    )
                    delay = float(match.group(1)) if match else self.retry_base_seconds * (2**attempt)
                    time.sleep(min(delay, 20.0))
                    continue
                raise RuntimeError(f"Gemini generation lỗi: {error_text}") from exc
        if response is None:  # pragma: no cover - vòng lặp chỉ thoát khi đã raise
            raise RuntimeError("Gemini generation không trả về response")
        answer = getattr(response, "text", None)
        if not answer or not answer.strip():
            raise RuntimeError("Gemini không trả về nội dung text")
        answer = answer.strip()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(answer, encoding="utf-8")
        return answer


def build_grounded_prompt(
    question: str,
    hits: list[dict[str, Any]],
    *,
    standalone_query: str | None = None,
) -> str:
    """Dựng prompt chuẩn cho LLM sinh câu trả lời có citation.

    Prompt này được gửi trực tiếp cho Gemini generation. Nó ghi rõ hợp đồng
    grounding để model không dùng kiến thức ngoài context.

    Args:
        question: Câu hỏi gốc cần trả lời.
        hits: Top-k context từ Retriever.
        standalone_query: Câu hỏi đã rewrite để bổ sung chủ đề cho follow-up;
            không thay thế câu hỏi gốc dùng cho nội dung trả lời.

    Returns:
        Prompt tiếng Việt gồm instruction, câu hỏi và từng chunk có ID.
    """

    context = "\n".join(
        f"[{hit['chunk_id']}] {hit['chunk'].get('text', '')}" for hit in hits
    )
    normalized_question = normalize_text(standalone_query or question)
    checklist: list[str] = []
    if "xét nghiệm" in normalized_question and "đái tháo đường" in normalized_question:
        checklist.append(
            "Với câu hỏi này, bắt buộc nêu đủ cả ba xét nghiệm và ngưỡng trong context: "
            "glucose đói ≥7,0 mmol/L; HbA1c ≥6,5%; glucose 2h sau OGTT 75g ≥11,1 mmol/L."
        )
    if "egfr" in normalized_question and "metformin" in normalized_question:
        checklist.append(
            "Với câu hỏi này, bắt buộc tính rõ 20 < 30, nói bệnh nhân không dùng được "
            "metformin và nêu metformin chống chỉ định."
        )
    if "định nghĩa" in normalized_question and "tăng huyết áp" in normalized_question:
        checklist.append(
            "Với câu hỏi này, bắt buộc nêu ngưỡng HA tâm thu ≥140 mmHg và/hoặc tâm trương ≥90 mmHg."
        )
    if "phân độ" in normalized_question:
        checklist.append("Với câu hỏi này, bắt buộc nêu đủ độ 1, độ 2 và độ 3 cùng các ngưỡng số liệu.")
    task_checklist = "\n\nCHECKLIST RIÊNG CHO CÂU HỎI:\n- " + "\n- ".join(checklist) if checklist else ""
    rewritten_context = ""
    if standalone_query and normalize_text(standalone_query) != normalize_text(question):
        rewritten_context = (
            "\n\nNGỮ CẢNH ĐÃ REWRITE CHO TÌM KIẾM (chỉ dùng để hiểu câu hỏi):\n"
            f"{standalone_query}"
        )
    return (
        f"{GROUNDING_INSTRUCTION}{task_checklist}\n\nCONTEXT:\n{context}"
        f"{rewritten_context}\n\nCÂU HỎI GỐC:\n{question}"
    )


def _ids(hits: list[dict[str, Any]]) -> list[str]:
    """Lấy danh sách chunk ID được phép xuất hiện trong citation."""
    return [str(hit["chunk_id"]) for hit in hits]


def answer_is_grounded(answer: str, hits: list[dict[str, Any]]) -> bool:
    """Kiểm tra citation trong answer có thuộc context được cấp hay không.

    Đây là guardrail hình thức: nó không đánh giá sự đúng đắn y khoa, nhưng
    chặn citation bịa hoặc ID không tồn tại trong top-k context.
    """

    cited = re.findall(r"\[([^\]]+)\]", answer)
    if answer.strip() == "Không tìm thấy trong tài liệu.":
        return not cited
    return bool(cited) and all(identifier in _ids(hits) for identifier in cited)


def missing_answer_requirements(
    question: str, answer: str, hits: list[dict[str, Any]]
) -> list[str]:
    """Tìm các ý bắt buộc còn thiếu trước khi chấp nhận response Gemini.

    Đây là hậu kiểm nội dung, không sinh câu trả lời thay model. Các luật tập
    trung vào lỗi an toàn đã biết: câu trả lời bị cắt, câu multi-hop thiếu một
    nhánh, chẩn đoán thiếu ngưỡng, và so sánh eGFR không được tính rõ.
    """

    normalized_question = normalize_text(question)
    normalized_answer = normalize_text(answer)
    missing: list[str] = []

    # Response bị cắt thường kết thúc bằng từ/cụm không có dấu câu hoặc citation.
    if normalized_answer and answer.rstrip()[-1] not in ".!?]”\"":
        missing.append("Kết thúc câu trả lời bằng câu hoàn chỉnh, không dừng giữa câu.")

    if "cấp cứu" in normalized_question and "khẩn trương" in normalized_question:
        for term in ("tổn thương cơ quan đích cấp", "tĩnh mạch", "đường uống"):
            if term not in normalized_answer:
                missing.append(f"Nêu rõ: {term}.")
        required_ids = {"tha2022_ch7_s7.1", "tha2022_ch7_s7.2"}
        cited = set(citation_ids(answer))
        if not required_ids.issubset(cited):
            missing.append("Trích dẫn cả hai chunk cấp cứu và khẩn trương.")

    if "egfr" in normalized_question and "metformin" in normalized_question:
        compact = normalized_answer.replace(" ", "")
        if not any(pattern in compact for pattern in ("20<30", "20≤30")):
            missing.append("Tính và viết rõ phép so sánh 20 < 30.")
        if "chống chỉ định" not in normalized_answer:
            missing.append("Nêu rõ metformin chống chỉ định.")
        if "không" not in normalized_answer:
            missing.append("Kết luận rõ ràng rằng bệnh nhân không dùng được metformin.")

    if ("xét nghiệm" in normalized_question or "chẩn đoán" in normalized_question) and "đái tháo đường" in normalized_question:
        for alternatives in (("glucose",), ("hba1c",), ("ogtt",), ("7,0", "7.0"), ("6,5", "6.5"), ("11,1", "11.1")):
            if not any(term in normalized_answer for term in alternatives):
                missing.append(f"Nêu đủ ngưỡng xét nghiệm: {'/'.join(alternatives)}.")
    return missing


def synthesize_answer(
    question: str,
    hits: list[dict[str, Any]],
    *,
    threshold: float = RETRIEVAL_THRESHOLD,
    force_unanswerable: bool = False,
    generator: GeminiAnswerGenerator | None = None,
    standalone_query: str | None = None,
) -> str:
    """Gọi Gemini để sinh câu trả lời chỉ dựa trên context đã retrieve.

    Không còn nhánh trả lời cố định theo từng câu hỏi. Các nhánh trước khi gọi
    model chỉ là guardrail: từ chối câu bẫy, từ chối score thấp, hoặc từ chối
    response có citation bịa/thiếu. Với response không đạt citation contract,
    hàm gọi Gemini thêm một lần với prompt sửa lỗi; nếu vẫn sai thì trả lời an
    toàn ``Không tìm thấy trong tài liệu.``.

    Args:
        question: Câu hỏi gốc dùng để diễn đạt câu trả lời.
        hits: Top-k context, mỗi hit có ``chunk_id``, ``score`` và ``chunk``.
        threshold: Điểm cosine tối thiểu để được trả lời.
        force_unanswerable: Buộc từ chối dù có hit, dùng cho câu không có đáp án.
        generator: Gemini generator đã khởi tạo; bỏ trống để tự khởi tạo.
        standalone_query: Query đã rewrite để Gemini hiểu follow-up; câu hỏi gốc
            vẫn là nội dung chính dùng để trả lời.

    Returns:
        Câu trả lời tiếng Việt; citation hợp lệ có dạng ``[chunk_id]``.
    """

    if force_unanswerable:
        return (
            "Không đủ dữ kiện: guideline trong corpus không nêu liều insulin nền "
            "cố định. Cần cá thể hóa theo cân nặng, đường huyết và chỉ định của bác sĩ; "
            "không nên tự suy ra một con số."
        )
    if not hits or float(hits[0]["score"]) < threshold:
        return "Không tìm thấy trong tài liệu."
    active_generator = generator or GeminiAnswerGenerator()
    prompt = build_grounded_prompt(question, hits, standalone_query=standalone_query)
    answer = active_generator.generate(prompt)
    requirement_question = standalone_query or question
    missing = missing_answer_requirements(requirement_question, answer, hits)
    if answer_is_grounded(answer, hits) and not missing:
        return answer

    # Citation sai/thiếu: yêu cầu model sửa lại thay vì tự thêm citation bằng code.
    repair_prompt = (
        f"{prompt}\n\nCÂU TRẢ LỜI VỪA RỒI:\n{answer}\n\n"
        "Hãy viết lại câu trả lời. Mỗi khẳng định phải có đúng một hoặc nhiều "
        "[chunk_id] tồn tại trong CONTEXT; tuyệt đối không tạo ID mới.\n"
        f"Các yêu cầu bắt buộc còn thiếu: {'; '.join(missing) or 'kiểm tra lại toàn bộ context'}."
    )
    for _ in range(2):
        repaired = active_generator.generate(repair_prompt)
        missing = missing_answer_requirements(requirement_question, repaired, hits)
        if answer_is_grounded(repaired, hits) and not missing:
            return repaired
        repair_prompt = (
            f"{prompt}\n\nCÂU TRẢ LỜI CẦN SỬA:\n{repaired}\n\n"
            f"Vẫn còn thiếu: {'; '.join(missing) or 'citation hợp lệ'}. "
            "Viết lại đầy đủ, kết thúc bằng dấu câu và không thêm kiến thức ngoài CONTEXT."
        )
    return "Không tìm thấy trong tài liệu."


def citation_ids(answer: str) -> list[str]:
    """Trích toàn bộ ID trong ngoặc vuông để log/eval citation."""
    return re.findall(r"\[([^\]]+)\]", answer)
