"""Validate data/eval/tickets.jsonl against the label guide and print a dataset report.

    python scripts/validate_tickets.py

Hard checks (exit code 1 on failure) are in rag_cskh.tickets.validate. The report also shows a retrieval
diagnostic: for tickets with gold evidence, whether plain BM25 over 120-word chunks finds at least one gold
section in its top 10. A ticket whose evidence BM25 cannot reach is not wrong, but it is worth re-reading,
and the numbers are the floor that Week-5 retrieval must beat.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.bm25 import BM25  # noqa: E402
from rag_cskh.chunking import chunk_corpus  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.text import tokens  # noqa: E402
from rag_cskh.tickets import load_tickets, validate  # noqa: E402

TYPES = ("normal", "missing_info", "no_evidence", "conflict")


def bm25_hits(tickets: list[dict], k: int = 10) -> dict[str, bool]:
    docs = [d for d in load_corpus()]
    chunks = chunk_corpus(docs, 120)
    bm25 = BM25([tokens(c.text) for c in chunks])
    hits = {}
    for t in tickets:
        if not t["gold_sections"]:
            continue
        ranked = list(dict.fromkeys(chunks[i].citation for i in bm25.rank(tokens(t["ticket"]))))[:k]
        hits[t["id"]] = bool(set(ranked) & set(t["gold_sections"]))
    return hits


def main() -> None:
    tickets = load_tickets()
    report = validate(tickets)
    print(f"{len(tickets)} tickets")
    print("\n| Type | Decision | Tickets | dev | test |\n|---|---|---:|---:|---:|")
    split = Counter((t["type"], t.get("split")) for t in tickets)
    for kind in TYPES:
        decision = next(t["expected_decision"] for t in tickets if t["type"] == kind)
        print(f"| {kind} | {decision} | {report.counts[kind]} | {split[(kind, 'dev')]} | {split[(kind, 'test')]} |")
    print("\n| Tag | Tickets |\n|---|---:|")
    for tag, n in report.tags.most_common():
        print(f"| {tag} | {n} |")

    gold_sizes = Counter(len(t["gold_sections"]) for t in tickets if t["gold_sections"])
    docs_cited = {g.split("#")[0] for t in tickets for g in t["gold_sections"]}
    print(f"\nGold sections per ticket: {dict(sorted(gold_sizes.items()))}; documents cited: {len(docs_cited)}")

    hits = bm25_hits(tickets)
    print("\nBM25 diagnostic (at least one gold section in top 10):")
    for kind in ("normal", "missing_info", "conflict"):
        ids = [t["id"] for t in tickets if t["type"] == kind]
        found = sum(hits[i] for i in ids)
        print(f"  {kind}: {found}/{len(ids)} ({found / len(ids):.0%})")
    misses = [i for i, ok in hits.items() if not ok]
    print(f"  not reached: {', '.join(misses) if misses else 'none'}")

    if report.errors:
        print("\nERRORS")
        print("\n".join(report.errors))
        sys.exit(1)
    print("\nAll label checks passed.")


if __name__ == "__main__":
    main()
