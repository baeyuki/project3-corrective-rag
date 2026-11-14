"""Shared answer prompt (Week 6), used unchanged by Vector RAG, Hybrid RAG and Corrective RAG.

The prompt restates the operating rules of the knowledge base itself, so every system is held to the same
standard and differences between systems come from the evidence they pass, not from the instructions:
- decisions follow QT-SU-DUNG-TAI-LIEU-2026 (answer / ask for missing information / escalate when no document
  has the information or two valid documents disagree for the customer's case);
- reply format follows QT-MAU-PHAN-HOI-2026 (greeting and restatement, main answer with conditions, next step
  with a time frame, closing; "em" / "anh/chị"; at most 120 words for chat; no document codes, no promises outside
  the documents, no estimates, never ask for passwords, PIN, CVV or OTP).
Citations go in a separate field, not in the reply text (customers must not see document codes); the
Week-8 citation check and the Week-9 judge read that field.

Output: OpenAI Structured Outputs with ``decision``, ``reply``, ``citations`` and ``missing_info``.
"""

from __future__ import annotations

import re

from rag_cskh.evidence import SectionStore

ANSWER_K = 5
DECISIONS = ("answer", "clarify", "escalate")

INSTRUCTIONS = """Bạn là trợ lý soạn phản hồi cho nhân viên chăm sóc khách hàng của một chuỗi bán lẻ điện máy tại Việt Nam. Hôm nay là thứ năm, 22/10/2026; ngày làm việc không gồm chủ nhật và ngày lễ.
Chỉ dùng các mục trong phần TÀI LIỆU (đều đang hiệu lực); không dùng kiến thức bên ngoài và không suy đoán.

Chọn một quyết định:
- "answer": tài liệu đủ để trả lời đúng trường hợp của khách. Trả lời kèm điều kiện áp dụng.
- "clarify": tài liệu có quy định nhưng cần thông tin riêng của khách mà ticket chưa có (ví dụ mã đơn, ngày giao, model cụ thể, kích thước) mới áp dụng được. Hỏi đúng thông tin còn thiếu, tối đa ba mục, kèm lý do cần.
- "escalate": không mục nào có thông tin khách hỏi, HOẶC hai tài liệu nêu khác nhau (thời hạn, mức phí, ngưỡng, điều kiện) cho đúng trường hợp của khách. Khi đó không tự chọn một bên, không cam kết; báo khách yêu cầu đã được chuyển bộ phận phụ trách và thời hạn phản hồi nếu tài liệu có nêu, không giải thích chi tiết nội bộ.
Hai mục nói về các thuộc tính khác nhau, hoặc một mục chỉ áp dụng cho phạm vi hẹp không gồm trường hợp của khách, thì không mâu thuẫn.

Quy chuẩn phản hồi (trường reply):
- Thứ tự: lời chào và xác nhận vấn đề; câu trả lời chính kèm điều kiện; bước tiếp theo (khách làm gì hoặc cửa hàng làm gì, có mốc thời gian nếu tài liệu có); lời kết.
- Xưng "em", gọi khách là "anh/chị". Câu ngắn, tối đa 120 từ.
- Không ghi mã tài liệu hay tên hệ thống nội bộ; không cam kết thời hạn, bồi thường hoặc ngoại lệ không có trong tài liệu; không đưa số liệu ước lượng; không yêu cầu mật khẩu, mã PIN, CVV hoặc OTP.

Trường citations: mã mục (dạng DOC-ID#section-id, đúng như trong ngoặc vuông) đã dùng làm căn cứ; khi escalate vì mâu thuẫn ghi cả hai mục; khi không có căn cứ để trống.
Trường missing_info: các thông tin hỏi khách (chỉ khi clarify, nếu không để trống)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "reply": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "missing_info": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "reply", "citations", "missing_info"],
    "additionalProperties": False,
}

_CITATION = re.compile(r"[A-Z0-9-]+#[a-z0-9-]+")


def build_messages(ticket: str, evidence: list[str], store: SectionStore) -> list[dict]:
    blocks = "\n\n".join(store.block(c) for c in evidence) or "(không có mục nào)"
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"TÀI LIỆU:\n{blocks}\n\nTICKET CỦA KHÁCH:\n{ticket}"}]


def clean_citations(raw: list[str], evidence: list[str]) -> tuple[list[str], list[str]]:
    """(valid, invalid): citations normalised to ``DOC#section``; valid = present in the context given."""
    found = list(dict.fromkeys(m for r in raw for m in _CITATION.findall(r)))
    allowed = set(evidence)
    return [c for c in found if c in allowed], [c for c in found if c not in allowed]


def generate(llm, ticket: str, evidence: list[str], store: SectionStore) -> dict:
    out, record = llm.chat_json(build_messages(ticket, evidence, store), SCHEMA, "answer")
    valid, invalid = clean_citations(out["citations"], evidence)
    return {**out, "citations": valid, "invalid_citations": invalid,
            "tokens": record["prompt_tokens"] + record["completion_tokens"], "latency_s": record["latency_s"]}
