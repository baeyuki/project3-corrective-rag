"""Ticket analysis (Corrective RAG step 1, Week 7).

One LLM call turns the raw ticket into:
- ``normalized``: the ticket rewritten with full Vietnamese diacritics and expanded abbreviations, same meaning;
- ``sub_questions``: 1-3 self-contained questions the customer actually asks, the main one first, each naming the
  product / service so it can be used as a retrieval query on its own; a question about the customer's own case
  stays about that case (v2, after the dev run: v1 added speculative follow-ups and turned "is it still possible
  for me?" into general questions);
- ``facts``: information the customer already gave (product, model, dates, amounts, distance, order code, ...);
- ``products``: product or model names mentioned.
The analysis only restates the ticket: it never sees documents or labels and must not add facts. If the call
returns no sub-question, the ticket itself is used (``Analysis.queries`` always contains the raw ticket).
"""

from __future__ import annotations

from dataclasses import dataclass, field

INSTRUCTIONS = """Bạn phân tích ticket của khách hàng gửi bộ phận chăm sóc khách hàng của một chuỗi bán lẻ điện máy tại Việt Nam. Hôm nay là thứ năm, 22/10/2026.
Chỉ diễn đạt lại nội dung ticket, không thêm thông tin, không trả lời câu hỏi.
- normalized: viết lại ticket bằng tiếng Việt có dấu đầy đủ, mở các từ viết tắt (vd "bh" -> "bảo hành", "ship" -> "giao hàng"), giữ nguyên ý và mọi con số.
- sub_questions: 1 đến 3 câu hỏi độc lập mà khách thực sự hỏi, câu chính đứng đầu. Mỗi câu phải tự hiểu được khi đứng riêng: nêu rõ sản phẩm, dịch vụ hoặc tình huống (vd "Hàng mua trong chương trình khuyến mãi tháng 10 được đổi do không ưng trong bao nhiêu ngày?"). Không tách nhỏ một câu hỏi đơn giản và không thêm câu hỏi khách không hỏi (vd "có cần gọi thợ không", "xử lý mất bao lâu"). Nếu khách hỏi về trường hợp của chính mình ("còn kịp không", "đơn của em", "em có được ... không"), câu chính phải giữ đúng trường hợp đó, không đổi thành câu hỏi chung.
- facts: các thông tin khách đã cung cấp (sản phẩm, model, ngày mua/giao, số tiền, khoảng cách, mã đơn, hạng thành viên, tình trạng sản phẩm...). Ghi ngắn gọn, đúng như khách nói; để trống nếu không có.
- products: tên sản phẩm hoặc model được nhắc tới, để trống nếu không có."""

SCHEMA = {
    "type": "object",
    "properties": {
        "normalized": {"type": "string"},
        "sub_questions": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {"type": "string"}},
        "products": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["normalized", "sub_questions", "facts", "products"],
    "additionalProperties": False,
}

MAX_SUB_QUESTIONS = 3


@dataclass
class Analysis:
    ticket: str
    normalized: str
    sub_questions: list[str]
    facts: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)

    @property
    def queries(self) -> list[str]:
        """Retrieval queries: raw ticket, normalized ticket, sub-questions (deduplicated, raw ticket first)."""
        out = [self.ticket, self.normalized, *self.sub_questions]
        seen, result = set(), []
        for q in out:
            key = " ".join(q.lower().split())
            if key and key not in seen:
                seen.add(key)
                result.append(q.strip())
        return result

    def to_dict(self) -> dict:
        return {"normalized": self.normalized, "sub_questions": self.sub_questions, "facts": self.facts,
                "products": self.products}


def build_messages(ticket: str) -> list[dict]:
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": f"TICKET:\n{ticket}"}]


def parse(ticket: str, out: dict) -> Analysis:
    subs = [s.strip() for s in out.get("sub_questions", []) if s and s.strip()][:MAX_SUB_QUESTIONS]
    normalized = (out.get("normalized") or "").strip() or ticket
    return Analysis(ticket, normalized, subs or [normalized],
                    [f.strip() for f in out.get("facts", []) if f.strip()],
                    [p.strip() for p in out.get("products", []) if p.strip()])


def analyze(llm, ticket: str) -> Analysis:
    out, _ = llm.chat_json(build_messages(ticket), SCHEMA, "ticket_analysis")
    return parse(ticket, out)
