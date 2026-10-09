"""Feasibility check: can an LLM grade retrieved evidence into the five verdicts the project needs?

For four toy tickets (one per expected outcome) it retrieves the top passages with BM25 from data/poc/toy_kb.jsonl
and asks the LLM for a verdict with OpenAI Structured Outputs. Cost: well under $0.01.

    python scripts/poc_grading.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_env import load_env  # noqa: E402
from rag_cskh.bm25 import BM25  # noqa: E402
from rag_cskh.text import tokens  # noqa: E402

TICKETS = [
    ("Tôi gửi bảo hành đã 20 ngày làm việc mà chưa xong, có được mượn máy dùng tạm không?", "sufficient"),
    ("Hủy đơn hàng giúp tôi với.", "missing_info"),
    ("Cửa hàng có cho thuê máy chiếu theo ngày không?", "no_evidence"),
    ("Hàng mua trong đợt khuyến mãi tháng 10 thì được đổi trả trong mấy ngày?", "conflict"),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["sufficient", "insufficient", "missing_info", "no_evidence", "conflict"]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "evidence_ids", "reason"],
    "additionalProperties": False,
}

SYSTEM = """You grade evidence for a customer-support assistant of a Vietnamese electronics retailer.
Use only the passages given. Expired passages are never evidence. Pick exactly one verdict:
- sufficient: active passages directly answer the ticket (list them in evidence_ids)
- insufficient: the topic is in scope but these passages do not contain the needed rule
- missing_info: the answer depends on information only the customer can give (e.g. order number)
- no_evidence: the request is outside what the documents cover
- conflict: two active passages from different documents give different values for the customer's case
Write `reason` in Vietnamese: it is shown to Vietnamese support agents."""


def main() -> int:
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing - run scripts/check_env.py first")
        return 1
    from openai import OpenAI
    client = OpenAI(timeout=60)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    docs = [json.loads(line) for line in (ROOT / "data" / "poc" / "toy_kb.jsonl").read_text(encoding="utf-8").splitlines() if line]
    bm25 = BM25([tokens(d["text"]) for d in docs])
    correct, total_tokens = 0, 0
    for ticket, expected in TICKETS:
        top = [docs[i] for i in bm25.rank(tokens(ticket))[:4]]
        evidence = "\n".join(f"[{d['id']}] ({d['status']}) {d['text']}" for d in top) or "(no passages found)"
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Ticket: {ticket}\n\nPassages:\n{evidence}"}],
            response_format={"type": "json_schema", "json_schema": {"name": "grade", "strict": True, "schema": SCHEMA}})
        out = json.loads(resp.choices[0].message.content)
        total_tokens += resp.usage.total_tokens
        correct += out["verdict"] == expected
        mark = "OK " if out["verdict"] == expected else "XX "
        print(f"{mark}expected={expected:12s} got={out['verdict']:12s} evidence={out['evidence_ids']}\n    {out['reason']}")
    print(f"\n{correct}/{len(TICKETS)} correct, {total_tokens} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
