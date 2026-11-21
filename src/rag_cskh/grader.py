"""Evidence grader (Corrective RAG step 3, Week 7).

The LLM judges, the code decides. One call receives the ticket, its analysis and the top ``K_GRADE`` reranked
sections (numbered 1..n, so no citation can be invented) and returns structured judgments:
- per section: relevant / partial / irrelevant;
- per sub-question: answered / needs_customer_info / not_in_documents, with the supporting sections;
- missing customer information, each item tied to the section that requires it;
- conflicts: two sections stating different values for the same attribute, and whether the customer's case is
  inside the scope of both.
A second, focused call (pairwise check) then examines every pair of relevant/partial sections from different
documents one by one: same attribute of the same service? does each section apply to this kind of case? what
outcome does each give for the customer's case, and do the outcomes differ? A pair is a conflict only if all hold.
This is the registry's definition: C02 (order 400,000 d, 5 km) is a conflict because one section makes it free and
the other does not; near-miss N01 (order >= 500,000 d within 10 km) is not, because both make it free.
``decide`` applies the rules of QT-SU-DUNG-TAI-LIEU-2026 deterministically, in this order:
1. a valid conflict -> escalate (two different documents, both graded relevant or partial, case in scope of both);
2. the main sub-question is not in the documents (or nothing is relevant) -> escalate;
3. the main sub-question needs customer information and at least one missing item is named -> clarify;
4. otherwise -> answer.
v1 -> v2 (dev run 2026-11-20): v1 asked the customer for information whenever any sub-question needed it, and the
single holistic call almost never filled the conflict list (1 of 8 conflict tickets, 0 of 6 registry probes,
although both sides were in the top 10). v2: only the main question can trigger clarify, a stricter definition
of "needs customer information", and the pairwise conflict check.
v2 -> v3 (same day): the v2 pair check asked whether the case was "in the scope" of each section; the model read
this as "does the customer qualify" and rejected 6 of 8 true conflicts (C01, C02, C03, K013, K016, K019). v3
separates "the section applies to this kind of case" from "the outcome for this case".
v3 -> v4 (same day): with gpt-4.1-mini, 19 of 23 pair-check conflicts were false, mostly complementary sections
(one adds steps or details the other omits) or identical outcomes. v4 requires outcomes that cannot both be true
and a concrete value on both sides; the code also rejects empty or identical outcomes.
Keeping the policy in code makes every decision traceable to a judgment and lets the judgments be tested and
tuned separately from the policy. The conflict check drops pairs the code can refute (same document, same section,
a side graded irrelevant, out-of-scope case), which is where near-misses are expected to be caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_cskh.analysis import Analysis
from rag_cskh.evidence import SectionStore

K_GRADE = 10
RELEVANCE = ("relevant", "partial", "irrelevant")
STATUS = ("answered", "needs_customer_info", "not_in_documents")

INSTRUCTIONS = """Bạn là bộ kiểm tra bằng chứng cho trợ lý chăm sóc khách hàng của một chuỗi bán lẻ điện máy tại Việt Nam. Hôm nay là thứ năm, 22/10/2026; ngày làm việc không gồm chủ nhật và ngày lễ. Mọi mục tài liệu được cung cấp đều đang hiệu lực. Không dùng kiến thức bên ngoài tài liệu.
Bạn KHÔNG soạn câu trả lời; bạn chỉ đánh giá các mục tài liệu so với ticket.

1. sections: với mỗi mục (theo số thứ tự), relevance là
   "relevant" = nêu trực tiếp quy định, giá trị, thông số hoặc thủ tục cần để xử lý trường hợp của khách;
   "partial" = có liên quan (điều kiện, ngoại lệ, bước xử lý) nhưng một mình chưa đủ;
   "irrelevant" = không giúp xử lý trường hợp này.
2. sub_questions: với mỗi câu hỏi con (theo số thứ tự, câu 1 là câu chính), status là
   "answered" = các mục tài liệu đủ để trả lời đúng trường hợp của khách với thông tin khách đã cung cấp;
   "needs_customer_info" = tài liệu có quy định, nhưng câu trả lời cho ĐÚNG câu khách hỏi thay đổi tùy theo thông tin riêng của khách mà ticket chưa có. Ví dụ: khách hỏi "còn kịp không" mà chưa nói ngày giao; tài liệu nêu giá trị khác nhau theo phiên bản, kích thước hoặc model mà khách chưa nói rõ; khách nhờ kiểm tra, hủy, sửa đơn mà chưa có mã đơn;
   "not_in_documents" = không mục nào nêu thông tin cần thiết (kể cả khi danh mục ghi "chưa có dữ liệu" hoặc không công bố, thông số không có trong danh mục, sản phẩm/dịch vụ không có trong tài liệu). Không hỏi khách thông tin mà chính tài liệu còn thiếu.
   evidence: số thứ tự các mục làm căn cứ.
   Câu hỏi chung về quy định, thông số, cách xử lý sự cố ("đổi trả trong mấy ngày?", "máy nén bảo hành mấy năm?", "cửa máy giặt không mở thì làm sao?") là "answered" khi tài liệu có câu trả lời. Thông tin chỉ cần cho bước xử lý sau (mã đơn để tra cứu, đặt lịch, xác minh bảo hành, giá mẫu mới) không làm câu hỏi thành "needs_customer_info".
3. missing_info: những thông tin riêng của khách cần hỏi (tối đa 3), mỗi mục kèm số thứ tự mục tài liệu yêu cầu thông tin đó. Chỉ điền khi có câu hỏi "needs_customer_info"; để trống nếu không.
4. conflicts: các cặp mục nêu giá trị KHÁC NHAU (thời hạn, mức phí, ngưỡng tiền, điều kiện) cho CÙNG một thuộc tính. Ghi thuộc tính, giá trị của từng bên, và in_scope_of_both = true chỉ khi trường hợp của khách nằm trong phạm vi áp dụng của CẢ HAI mục (đúng sản phẩm, chương trình, kênh bán, khoảng thời gian, khoảng giá trị). Không phải mâu thuẫn: hai mục nói về thuộc tính khác nhau (vd thời hạn bảo hành của hãng và thời gian xử lý của cửa hàng); một mục chỉ áp dụng cho phạm vi hẹp không gồm trường hợp của khách; ưu đãi riêng của kênh bán của hãng; hai mục cho cùng kết quả cho trường hợp của khách. Để trống nếu không có.
5. reason: một hai câu tiếng Việt tóm tắt đánh giá."""

SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "relevance": {"type": "string", "enum": list(RELEVANCE)}},
            "required": ["id", "relevance"], "additionalProperties": False}},
        "sub_questions": {"type": "array", "items": {
            "type": "object",
            "properties": {"index": {"type": "integer"}, "status": {"type": "string", "enum": list(STATUS)},
                           "evidence": {"type": "array", "items": {"type": "integer"}}},
            "required": ["index", "status", "evidence"], "additionalProperties": False}},
        "missing_info": {"type": "array", "items": {
            "type": "object",
            "properties": {"item": {"type": "string"}, "required_by": {"type": "integer"}},
            "required": ["item", "required_by"], "additionalProperties": False}},
        "conflicts": {"type": "array", "items": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "attribute": {"type": "string"},
                           "value_a": {"type": "string"}, "value_b": {"type": "string"},
                           "in_scope_of_both": {"type": "boolean"}},
            "required": ["a", "b", "attribute", "value_a", "value_b", "in_scope_of_both"], "additionalProperties": False}},
        "reason": {"type": "string"},
    },
    "required": ["sections", "sub_questions", "missing_info", "conflicts", "reason"],
    "additionalProperties": False,
}

PAIR_INSTRUCTIONS = """Bạn kiểm tra mâu thuẫn giữa các mục tài liệu nội bộ của một chuỗi bán lẻ điện máy tại Việt Nam, đối với ticket của khách. Hôm nay là 22/10/2026. Mọi mục đều đang hiệu lực.
Với TỪNG cặp mục được liệt kê, trả lời độc lập:
- same_attribute: cả hai mục cùng quy định MỘT thuộc tính của CÙNG một loại dịch vụ/tình huống, dùng để trả lời câu hỏi chính (vd cùng là thời hạn đổi hàng do nhu cầu; cùng là điều kiện miễn phí giao hàng; cùng là thời gian xử lý bảo hành tại cửa hàng). Khác loại dịch vụ (giao hàng và lắp đặt) hoặc khác thuộc tính (thời hạn bảo hành của hãng và thời gian xử lý của cửa hàng) thì là false. Nếu một trong hai mục không nêu giá trị cụ thể cho thuộc tính đó (chỉ nói chung, chỉ mô tả quy trình, hoặc không đề cập) thì là false.
- attribute, value_a, value_b: tên thuộc tính và quy định mỗi mục nêu (ngắn gọn, giữ con số); để trống nếu same_attribute là false.
- applies_a / applies_b: mục đó nói về đúng loại trường hợp của khách: đúng chương trình, sản phẩm, dịch vụ, kênh bán (cửa hàng hay kênh riêng của hãng), loại khách (cá nhân hay doanh nghiệp), thời điểm. KHÔNG xét khách có đạt điều kiện hay ngưỡng của mục hay không; việc đó thuộc outcome. Một mục áp dụng chung cho mọi đơn của cửa hàng thì applies là true.
- outcome_a / outcome_b: nếu chỉ dựa vào riêng mục đó, kết quả cho trường hợp của khách là gì (vd "được miễn phí giao", "phải trả phí giao", "đổi trong 15 ngày, miễn phí", "xử lý tối đa 10 ngày làm việc"). Nếu khách hỏi chung, kết quả là giá trị mục nêu.
- outcomes_differ: true CHỈ KHI hai kết quả không thể cùng đúng: làm theo mục a thì trả lời khách khác với làm theo mục b (khác số ngày, khác mức phí, một bên được miễn phí còn bên kia phải trả, một bên đủ điều kiện còn bên kia không). Là false khi: hai mục bổ sung cho nhau (một mục nêu thêm bước, chi tiết hoặc ngoại lệ mà mục kia không nhắc tới); khác cách diễn đạt; hai mục có ngưỡng khác nhau nhưng cho cùng kết quả với trường hợp của khách."""

PAIR_SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "items": {
        "type": "object",
        "properties": {"pair": {"type": "integer"}, "same_attribute": {"type": "boolean"}, "attribute": {"type": "string"},
                       "value_a": {"type": "string"}, "value_b": {"type": "string"},
                       "applies_a": {"type": "boolean"}, "applies_b": {"type": "boolean"},
                       "outcome_a": {"type": "string"}, "outcome_b": {"type": "string"},
                       "outcomes_differ": {"type": "boolean"}},
        "required": ["pair", "same_attribute", "attribute", "value_a", "value_b", "applies_a", "applies_b", "outcome_a",
                     "outcome_b", "outcomes_differ"],
        "additionalProperties": False}}},
    "required": ["pairs"],
    "additionalProperties": False,
}

MAX_PAIR_SECTIONS = 5


@dataclass
class Grade:
    decision: str                                   # answer | clarify | escalate
    trigger: str                                    # conflict | no_evidence | missing_info | sufficient
    relevance: dict[str, str]                       # citation -> relevant | partial | irrelevant
    sub_questions: list[dict]                       # {question, status, evidence: [citations]}
    missing_info: list[dict]                        # {item, required_by: citation}
    conflicts: list[dict]                           # valid conflicts, citations instead of numbers
    rejected_conflicts: list[dict] = field(default_factory=list)   # flagged by the LLM, refuted by the code
    reason: str = ""
    pair_checks: list[dict] = field(default_factory=list)          # v2 pairwise verification, all pairs

    @property
    def evidence(self) -> list[str]:
        """Sections to pass to the answer step: relevant first, then partial, in reranked order."""
        rel = [c for c, r in self.relevance.items() if r == "relevant"]
        return rel + [c for c, r in self.relevance.items() if r == "partial"]

    def to_dict(self) -> dict:
        return {"decision": self.decision, "trigger": self.trigger, "relevance": self.relevance,
                "sub_questions": self.sub_questions, "missing_info": self.missing_info, "conflicts": self.conflicts,
                "rejected_conflicts": self.rejected_conflicts, "reason": self.reason, "evidence": self.evidence,
                "pair_checks": self.pair_checks}


def build_messages(analysis: Analysis, citations: list[str], store: SectionStore) -> list[dict]:
    blocks = "\n\n".join(store.block(c, str(i)) for i, c in enumerate(citations, 1)) or "(không có mục nào)"
    subs = "\n".join(f"{i}. {q}" for i, q in enumerate(analysis.sub_questions, 1))
    facts = "; ".join(analysis.facts) or "(không có)"
    user = (f"TICKET CỦA KHÁCH:\n{analysis.ticket}\n\nCÂU HỎI CON:\n{subs}\n\nTHÔNG TIN KHÁCH ĐÃ CUNG CẤP: {facts}"
            f"\n\nCÁC MỤC TÀI LIỆU:\n{blocks}")
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]


def _cit(citations: list[str], i) -> str | None:
    return citations[i - 1] if isinstance(i, int) and 1 <= i <= len(citations) else None


def interpret(out: dict, analysis: Analysis, citations: list[str]) -> Grade:
    """Map the LLM output (numbers) to citations and apply the decision rules."""
    relevance = {c: "irrelevant" for c in citations}
    for s in out.get("sections", []):
        c = _cit(citations, s.get("id"))
        if c and s.get("relevance") in RELEVANCE:
            relevance[c] = s["relevance"]

    by_index = {q.get("index"): q for q in out.get("sub_questions", [])}
    subs = []
    for i, question in enumerate(analysis.sub_questions, 1):
        q = by_index.get(i, {})
        status = q.get("status") if q.get("status") in STATUS else "not_in_documents"
        subs.append({"question": question, "status": status,
                     "evidence": [c for c in (_cit(citations, e) for e in q.get("evidence", [])) if c]})

    missing = []
    for m in out.get("missing_info", []):
        item = (m.get("item") or "").strip()
        if item:
            missing.append({"item": item, "required_by": _cit(citations, m.get("required_by"))})

    valid, rejected = [], []
    for x in out.get("conflicts", []):
        a, b = _cit(citations, x.get("a")), _cit(citations, x.get("b"))
        row = {"a": a, "b": b, "attribute": x.get("attribute", ""), "value_a": x.get("value_a", ""),
               "value_b": x.get("value_b", ""), "in_scope_of_both": bool(x.get("in_scope_of_both"))}
        why = None
        if not a or not b or a == b:
            why = "invalid section numbers"
        elif a.split("#")[0] == b.split("#")[0]:
            why = "same document"
        elif relevance[a] == "irrelevant" or relevance[b] == "irrelevant":
            why = "a side graded irrelevant"
        elif not row["in_scope_of_both"]:
            why = "customer case not in scope of both"
        elif " ".join(row["value_a"].lower().split()) == " ".join(row["value_b"].lower().split()):
            why = "same value"
        (rejected if why else valid).append({**row, **({"rejected_because": why} if why else {})})

    g = Grade("", "", relevance, subs, missing, valid, rejected, out.get("reason", ""))
    decide(g)
    return g


def decide(g: Grade) -> Grade:
    """Rules of QT-SU-DUNG-TAI-LIEU-2026, in priority order; only the main (first) sub-question can clarify/escalate."""
    any_relevant = any(r != "irrelevant" for r in g.relevance.values())
    main = g.sub_questions[0]["status"] if g.sub_questions else "not_in_documents"
    if g.conflicts:
        g.decision, g.trigger = "escalate", "conflict"
    elif main == "not_in_documents" or not any_relevant:
        g.decision, g.trigger = "escalate", "no_evidence"
    elif main == "needs_customer_info" and g.missing_info:
        g.decision, g.trigger = "clarify", "missing_info"
    else:
        g.decision, g.trigger = "answer", "sufficient"
    return g


def candidate_pairs(g: Grade, citations: list[str]) -> list[tuple[str, str]]:
    """Pairs of relevant/partial sections from different documents (first MAX_PAIR_SECTIONS in reranked order)."""
    kept = [c for c in citations if g.relevance.get(c) in ("relevant", "partial")][:MAX_PAIR_SECTIONS]
    return [(a, b) for i, a in enumerate(kept) for b in kept[i + 1:] if a.split("#")[0] != b.split("#")[0]]


def build_pair_messages(analysis: Analysis, pairs: list[tuple[str, str]], store: SectionStore) -> list[dict]:
    secs = list(dict.fromkeys(c for p in pairs for c in p))
    num = {c: i for i, c in enumerate(secs, 1)}
    blocks = "\n\n".join(store.block(c, str(num[c])) for c in secs)
    listing = "\n".join(f"Cặp {k}: mục {num[a]} (a) và mục {num[b]} (b)" for k, (a, b) in enumerate(pairs, 1))
    facts = "; ".join(analysis.facts) or "(không có)"
    user = (f"TICKET CỦA KHÁCH:\n{analysis.ticket}\n\nCÂU HỎI CHÍNH: {analysis.sub_questions[0]}\n"
            f"THÔNG TIN KHÁCH ĐÃ CUNG CẤP: {facts}\n\nCÁC MỤC TÀI LIỆU:\n{blocks}\n\nCÁC CẶP CẦN KIỂM TRA:\n{listing}")
    return [{"role": "system", "content": PAIR_INSTRUCTIONS}, {"role": "user", "content": user}]


def apply_pair_checks(g: Grade, pairs: list[tuple[str, str]], out: dict) -> Grade:
    """A pair is a conflict only if every focused judgment holds; the code keeps all checks for inspection."""
    known = {frozenset((x["a"], x["b"])) for x in g.conflicts}
    for x in out.get("pairs", []):
        k = x.get("pair")
        if not isinstance(k, int) or not 1 <= k <= len(pairs):
            continue
        a, b = pairs[k - 1]
        row = {"a": a, "b": b, **{f: x.get(f) for f in ("same_attribute", "attribute", "value_a", "value_b", "applies_a",
                                                          "applies_b", "outcome_a", "outcome_b", "outcomes_differ")}}
        oa, ob = (" ".join((row[k] or "").lower().split()) for k in ("outcome_a", "outcome_b"))
        va, vb = ((row[k] or "").strip() for k in ("value_a", "value_b"))
        row["conflict"] = bool(row["same_attribute"] and row["applies_a"] and row["applies_b"] and row["outcomes_differ"]
                               and oa and ob and oa != ob and va and vb)
        g.pair_checks.append(row)
        if row["conflict"] and frozenset((a, b)) not in known:
            g.conflicts.append({"a": a, "b": b, "attribute": row["attribute"], "value_a": row["outcome_a"] or row["value_a"],
                                "value_b": row["outcome_b"] or row["value_b"], "in_scope_of_both": True, "source": "pair_check"})
            known.add(frozenset((a, b)))
    return decide(g)


def grade(llm, analysis: Analysis, citations: list[str], store: SectionStore, pairwise: bool = True) -> Grade:
    if not citations:
        return Grade("escalate", "no_evidence", {}, [{"question": q, "status": "not_in_documents", "evidence": []}
                                                     for q in analysis.sub_questions], [], [], [], "no candidates")
    out, _ = llm.chat_json(build_messages(analysis, citations, store), SCHEMA, "evidence_grade")
    g = interpret(out, analysis, citations)
    if pairwise:
        pairs = candidate_pairs(g, citations)
        if pairs:
            pout, _ = llm.chat_json(build_pair_messages(analysis, pairs, store), PAIR_SCHEMA, "conflict_pairs")
            apply_pair_checks(g, pairs, pout)
    return g
