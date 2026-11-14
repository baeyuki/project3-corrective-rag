"""LLM reranker over the hybrid candidates (Week 6).

Pointwise relevance grades for all candidates in one call (listwise context, pointwise output): the model sees the
ticket and the top-``depth`` sections numbered 1..n and returns a grade 0-3 for each. Sections are re-ordered by
grade, ties keep the first-stage (hybrid RRF) order, and a candidate the model forgets keeps its first-stage
position behind the graded ones of the same grade -1 (it is never dropped). Numbered ids instead of citations keep
the output short and make invented citations impossible.

The instruction asks to grade every section on its own and NOT to choose between sections that state different
values: a reranker that resolves contradictions would hide the conflicts the Corrective-RAG grader must detect.
No label, gold section or conflict registry is used.
"""

from __future__ import annotations

from rag_cskh.evidence import SectionStore

RERANK_DEPTH = 20

INSTRUCTIONS = """Bạn chấm mức liên quan của các mục tài liệu nội bộ đối với một ticket của khách hàng gửi bộ phận chăm sóc khách hàng (chuỗi bán lẻ điện máy tại Việt Nam). Ticket có thể viết không dấu.
Chấm từng mục độc lập theo thang:
3 = mục nêu trực tiếp quy định, giá trị, thông số hoặc thủ tục cần để xử lý đúng trường hợp của khách (kể cả mục nêu những thông tin khách phải cung cấp);
2 = mục nêu điều kiện, ngoại lệ hoặc bước xử lý liên quan trực tiếp, nhưng một mình chưa đủ;
1 = cùng chủ đề nhưng không giúp xử lý trường hợp này;
0 = không liên quan.
Không chọn giữa các mục: nếu nhiều mục cùng nói về điều khách hỏi, kể cả khi chúng nêu giá trị khác nhau, chấm mỗi mục theo mức liên quan của nó. Chú ý phạm vi áp dụng ghi trong mục (sản phẩm, model, chương trình, kênh bán, thời gian).
Trả về điểm cho mọi mục, theo số thứ tự của mục."""

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "score": {"type": "integer", "enum": [0, 1, 2, 3]}},
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def build_messages(ticket: str, citations: list[str], store: SectionStore) -> list[dict]:
    blocks = "\n\n".join(store.block(c, str(i)) for i, c in enumerate(citations, 1))
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"TICKET CỦA KHÁCH:\n{ticket}\n\nCÁC MỤC TÀI LIỆU:\n{blocks}"}]


def order_by_scores(citations: list[str], scores: list[dict]) -> tuple[list[str], dict[str, int]]:
    graded: dict[str, int] = {}
    for item in scores:
        i = item.get("id")
        if isinstance(i, int) and 1 <= i <= len(citations) and citations[i - 1] not in graded:
            graded[citations[i - 1]] = int(item["score"])
    grade = {c: graded.get(c, -1) for c in citations}
    ranked = sorted(citations, key=lambda c: (-grade[c], citations.index(c)))
    return ranked, grade


class LLMReranker:
    name = "llm-rerank"

    def __init__(self, llm, store: SectionStore, depth: int = RERANK_DEPTH):
        self.llm, self.store, self.depth = llm, store, depth

    def rerank(self, ticket: str, citations: list[str]) -> tuple[list[str], dict[str, int]]:
        head, tail = citations[:self.depth], citations[self.depth:]
        if not head:
            return [], {}
        out, _ = self.llm.chat_json(build_messages(ticket, head, self.store), SCHEMA, "rerank")
        ranked, grade = order_by_scores(head, out["scores"])
        return ranked + tail, grade
