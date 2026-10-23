"""LLM cross-check of the ticket labels (Week 4 quality control).

    python scripts/llm_label_check.py                 # all 168 tickets, JUDGE_MODEL from .env (default gpt-4.1-mini)
    python scripts/llm_label_check.py --limit 5       # quick trial on 5 tickets
    python scripts/llm_label_check.py --report-only   # rebuild the report from cached answers, no API calls

Blind labelling: for each ticket the model receives
  - the decision procedure of docs/week4_label_guide.md (section 2, without the conflict registry and the list of
    near-misses, which would give answers away),
  - a table of contents of every document valid on 2026-10-22 (titles and section headings, so it knows what the KB
    covers and can say that something is absent),
  - the full text of the candidate sections retrieved for the ticket (union of BM25 top 10 and vector top 10, or
    BM25 top 20 with --no-dense),
  - the ticket text.
It never sees the reference type, gold sections, key facts or reference reply. It answers with OpenAI Structured
Outputs: type, evidence sections, missing slots and a short reason in Vietnamese.

Outputs (reports/week4/, git-ignored): llm_labels.jsonl (cache: one answer per ticket, reused on re-runs) and
llm_label_check.md (agreement, Cohen's kappa, confusion matrix, evidence overlap, list of disagreements).
Cost: about 10k input tokens per ticket (the table of contents is a fixed prefix that OpenAI caches); roughly
USD 0.5-1 for 168 tickets with gpt-4.1-mini. Run --limit 5 first to check the setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.chunking import chunk_corpus  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.embeddings import load_env  # noqa: E402
from rag_cskh.retrieval import BM25Retriever, DenseRetriever  # noqa: E402
from rag_cskh.tickets import SNAPSHOT, cohen_kappa, load_tickets  # noqa: E402

OUT = ROOT / "reports" / "week4"
TYPES = ("normal", "missing_info", "no_evidence", "conflict")

INSTRUCTIONS = """Bạn là người gán nhãn độc lập cho bộ đánh giá của trợ lý chăm sóc khách hàng (CSKH) của một chuỗi bán lẻ điện máy tại Việt Nam.
Mọi ticket được gửi ngày 22/10/2026 (thứ năm). Chỉ dùng các tài liệu được cung cấp (đều đang hiệu lực vào ngày này); không dùng kiến thức bên ngoài.
Ngày làm việc không gồm chủ nhật và ngày lễ.

Trả lời lần lượt các câu hỏi; câu "có" đầu tiên quyết định nhãn:
1. Có hai tài liệu đang hiệu lực nêu giá trị KHÁC NHAU (thời hạn, mức phí, ngưỡng tiền, điều kiện) cho ĐÚNG trường hợp của khách không? -> "conflict".
   Chỉ là conflict nếu trường hợp của khách nằm trong phạm vi áp dụng của cả hai tài liệu và hai tài liệu cho kết quả khác nhau. Hai tài liệu nói về thuộc tính khác nhau, hoặc một tài liệu có phạm vi hẹp không bao gồm trường hợp của khách, thì KHÔNG phải conflict.
2. Không tài liệu nào chứa thông tin cần để trả lời? -> "no_evidence". Gồm: giá trị mà danh mục ghi là chưa có dữ liệu/không công bố, giá bán và tồn kho hiện tại, dịch vụ không có trong tài liệu hoặc được ghi là ngoài phạm vi, sản phẩm không có trong danh mục.
3. Câu trả lời phụ thuộc thông tin riêng của khách mà ticket chưa cung cấp, và tài liệu có nêu cần thông tin đó? -> "missing_info". Ví dụ: mã đơn trước khi thao tác trên đơn, đúng phiên bản model khi các phiên bản khác nhau, ngày giao khi phải kiểm tra thời hạn.
4. Còn lại -> "normal" (trả lời được từ tài liệu).
Quy tắc phân định:
- Câu hỏi chung về một quy định ("đổi trả trong mấy ngày?") là normal; chỉ là missing_info khi khách nhờ xử lý/đánh giá trường hợp của mình mà thiếu thông tin bắt buộc.
- Nếu ticket đã có đủ chi tiết để quyết định (ngày, số tiền, khoảng cách, model) thì là normal dù nhân viên vẫn cần mã đơn để thực hiện.
- Dấu hiệu an toàn (khói, mùi khét, tia lửa, pin phồng, nước gần điện) vẫn gán theo thủ tục trên, thường là normal.

Trả về:
- type: một trong normal, missing_info, no_evidence, conflict;
- evidence: mã mục "DOC-ID#section-id" làm căn cứ (normal: các mục cần để trả lời; conflict: hai mục mâu thuẫn; missing_info: mục nêu thông tin cần có; no_evidence: để trống). Chỉ dùng mã có trong mục lục;
- missing_slots: thông tin cần hỏi khách (chỉ với missing_info);
- reason: một hai câu tiếng Việt giải thích."""

SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(TYPES)},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "missing_slots": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["type", "evidence", "missing_slots", "reason"],
    "additionalProperties": False,
}


def table_of_contents(docs) -> str:
    lines = []
    for d in docs:
        if d.is_active(SNAPSHOT):
            lines.append(f"{d.doc_id} | {d.title}")
            lines += [f"  #{s.section_id}: {s.heading}" for s in d.sections]
    return "\n".join(lines)


class Candidates:
    def __init__(self, docs, dense: bool):
        self.sections = {s.citation: (d.title, s) for d in docs for s in d.sections}
        chunks = chunk_corpus(docs, 120)
        self.retrievers = [BM25Retriever(chunks, active_on=SNAPSHOT)]
        self.depth = 20
        if dense:
            from rag_cskh.embeddings import openai_embedder
            self.retrievers.append(DenseRetriever(chunks, openai_embedder(), SNAPSHOT))
            self.depth = 10

    def citations(self, ticket: str) -> list[str]:
        out: list[str] = []
        for r in self.retrievers:
            for c in r.search_sections(ticket, self.depth):
                if c not in out:
                    out.append(c)
        return out

    def text(self, citations: list[str]) -> str:
        blocks = []
        for c in citations:
            title, s = self.sections[c]
            blocks.append(f"[{c}] {title} > {s.heading}\n{s.body}")
        return "\n\n".join(blocks)


def build_messages(ticket_text: str, toc: str, candidates_text: str) -> list[dict]:
    """The only inputs the model sees. Reference labels are never passed here."""
    return [
        {"role": "system", "content": INSTRUCTIONS + "\n\nMỤC LỤC TÀI LIỆU ĐANG HIỆU LỰC:\n" + toc},
        {"role": "user", "content": f"NỘI DUNG CÁC MỤC LIÊN QUAN:\n{candidates_text}\n\nTICKET CỦA KHÁCH:\n{ticket_text}"},
    ]


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["key"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def write_report(tickets: list[dict], answers: dict[str, dict], model: str) -> str:
    rows = [(t, answers[t["id"]]) for t in tickets if t["id"] in answers]
    ref = [t["type"] for t, _ in rows]
    llm = [a["type"] for _, a in rows]
    agree = sum(x == y for x, y in zip(ref, llm))
    lines = [f"# LLM cross-check of ticket labels ({model}, {len(rows)} tickets)\n",
             f"Type agreement: {agree}/{len(rows)} = {agree / len(rows):.2f}; Cohen's kappa = {cohen_kappa(ref, llm):.2f}\n",
             "Rows = reference label, columns = LLM label\n",
             "| | " + " | ".join(TYPES) + " | agreement |", "|---|" + "---:|" * (len(TYPES) + 1)]
    matrix = Counter(zip(ref, llm))
    for a in TYPES:
        n = sum(matrix[(a, b)] for b in TYPES)
        lines.append(f"| {a} | " + " | ".join(str(matrix[(a, b)]) for b in TYPES) + f" | {matrix[(a, a)]}/{n} |")

    hits = [bool(set(t["gold_sections"]) & set(a["evidence"])) for t, a in rows if t["type"] == a["type"] == "normal"]
    both = [set(t["gold_sections"][:1]) <= set(a["evidence"]) and bool(set(t["gold_sections"][1:]) & set(a["evidence"]))
            for t, a in rows if t["type"] == a["type"] == "conflict"]
    if hits:
        lines.append(f"\nNormal tickets labelled normal: the LLM cites at least one gold section in {sum(hits)}/{len(hits)}.")
    if both:
        lines.append(f"Conflict tickets labelled conflict: the LLM cites both registered sides in {sum(both)}/{len(both)}.")

    tags = Counter(tag for t, a in rows if t["type"] != a["type"] for tag in t["tags"])
    if tags:
        lines.append("\nTags of disagreeing tickets: " + ", ".join(f"{k} {v}" for k, v in tags.most_common()))
    lines.append("\n## Disagreements to adjudicate\n")
    lines.append("| Id | Reference | LLM | Ticket | LLM evidence | LLM reason |\n|---|---|---|---|---|---|")
    for t, a in rows:
        if t["type"] != a["type"]:
            text = t["ticket"].replace("|", "/")
            lines.append(f"| {t['id']} | {t['type']} | {a['type']} | {text} | {' '.join(a['evidence'])} | {a['reason'].replace('|', '/')} |")
    usage = sum(a.get("tokens", 0) for _, a in rows)
    lines.append(f"\nTokens used (all cached answers): {usage:,}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only the first N tickets (trial run)")
    parser.add_argument("--no-dense", action="store_true", help="BM25-only candidates (no embedding calls)")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    load_env()
    model = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
    tickets = load_tickets()
    if args.limit:
        tickets = tickets[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "llm_labels.jsonl"
    cache = load_cache(cache_path)

    docs = load_corpus()
    toc = table_of_contents(docs)
    answers: dict[str, dict] = {}
    if not args.report_only:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY missing - run scripts/check_env.py first")
            return 1
        from openai import OpenAI
        client = OpenAI(timeout=120)
        cands = Candidates(docs, dense=not args.no_dense)
        with cache_path.open("a", encoding="utf-8") as f:
            for i, t in enumerate(tickets, 1):
                cits = cands.citations(t["ticket"])
                messages = build_messages(t["ticket"], toc, cands.text(cits))
                key = t["id"] + ":" + hashlib.sha256((model + json.dumps(messages, ensure_ascii=False)).encode()).hexdigest()[:16]
                if key in cache:
                    answers[t["id"]] = cache[key]
                    continue
                resp = client.chat.completions.create(
                    model=model, temperature=0, messages=messages,
                    response_format={"type": "json_schema", "json_schema": {"name": "label", "strict": True, "schema": SCHEMA}})
                out = json.loads(resp.choices[0].message.content)
                row = {"key": key, "id": t["id"], **out, "candidates": cits, "tokens": resp.usage.total_tokens}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                answers[t["id"]] = row
                print(f"{i:3d}/{len(tickets)} {t['id']} ref={t['type']:12s} llm={out['type']}")
    else:
        latest = {}
        for row in cache.values():
            latest[row["id"]] = row
        answers = latest

    report = write_report(tickets, answers, model)
    (OUT / "llm_label_check.md").write_text(report, encoding="utf-8")
    print("\n" + report.split("## Disagreements")[0])
    print(f"Full report: {OUT / 'llm_label_check.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
