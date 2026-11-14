"""Automatic answer-level metrics (Week 6; the LLM judge for correctness and faithfulness comes in Week 9).

Per ticket:
- decision_ok      predicted decision == expected_decision (answer / clarify / escalate).
- fact_cov         normal tickets: share of ``key_facts`` found in the reply. A fact counts as found when at least
                   75% of its content syllables (accent-folded, stopwords removed) occur in the reply. This is a
                   lexical approximation: it misses paraphrases and ignores word order, so it is reported as a
                   screening signal, not as correctness.
- slot_cov         missing_info tickets: same test for ``missing_slots`` (text in parentheses ignored) against
                   reply + missing_info.
- context_ok       the evidence passed to the prompt contains the gold: all gold sections (normal), both sides
                   (conflict), at least one gold section (missing_info). Separates retrieval failures from
                   generation failures.
- cited_ok         normal: a gold section is cited; conflict: both sides are cited.
- invalid_cit      citations not present in the context (the model invented or mistyped them).
- code_leak        the reply shows a document code to the customer (forbidden by QT-MAU-PHAN-HOI-2026).
- words / over_120 reply length in whitespace words; over the 120-word chat limit.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean

from rag_cskh.answer import DECISIONS
from rag_cskh.text import STOPWORDS, fold, syllables

COVER = 0.75
_FOLDED_STOP = {fold(w) for w in STOPWORDS}
_CODE = re.compile(r"\b(?:CS|QT|FAQ|HD|SP|NSX|KM|TB|BG|HT)-[A-Z0-9-]{3,}")


def content_syllables(text: str) -> list[str]:
    return [s for s in (fold(x) for x in syllables(text)) if s not in _FOLDED_STOP]


def covered(phrase: str, text: str, threshold: float = COVER) -> bool:
    need = set(content_syllables(re.sub(r"\([^)]*\)", " ", phrase))) or set(content_syllables(phrase))
    if not need:
        return True
    have = set(content_syllables(text))
    return len(need & have) / len(need) >= threshold


def context_ok(ticket: dict, evidence: list[str]) -> float | None:
    gold, ev = ticket["gold_sections"], set(evidence)
    if not gold:
        return None
    if ticket["type"] == "conflict":
        return float(gold[0] in ev and bool(ev & set(gold[1:])))
    if ticket["type"] == "missing_info":
        return float(bool(ev & set(gold)))
    return float(set(gold) <= ev)


def ticket_answer_metrics(ticket: dict, evidence: list[str], answer: dict) -> dict:
    reply = answer["reply"]
    m = {"decision_ok": float(answer["decision"] == ticket["expected_decision"]),
         "context_ok": context_ok(ticket, evidence),
         "invalid_cit": len(answer.get("invalid_citations", [])),
         "code_leak": float(bool(_CODE.search(reply))),
         "words": len(reply.split()),
         "fact_cov": None, "slot_cov": None, "cited_ok": None}
    m["over_120"] = float(m["words"] > 120)
    cited = set(answer["citations"])
    gold = ticket["gold_sections"]
    if ticket["type"] == "normal":
        if ticket["key_facts"]:
            m["fact_cov"] = mean(covered(f, reply) for f in ticket["key_facts"])
        m["cited_ok"] = float(bool(cited & set(gold)))
    elif ticket["type"] == "conflict":
        m["cited_ok"] = float(gold[0] in cited and bool(cited & set(gold[1:])))
    elif ticket["type"] == "missing_info" and ticket["missing_slots"]:
        asked = reply + " " + " ".join(answer["missing_info"])
        m["slot_cov"] = mean(covered(s, asked) for s in ticket["missing_slots"])
    return m


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def macro_f1(gold: list[str], pred: list[str], labels=DECISIONS) -> float:
    f1s = []
    for c in labels:
        tp = sum(g == p == c for g, p in zip(gold, pred))
        fp = sum(p == c and g != c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return mean(f1s)


def confusion(gold: list[str], pred: list[str]) -> Counter:
    return Counter(zip(gold, pred))
