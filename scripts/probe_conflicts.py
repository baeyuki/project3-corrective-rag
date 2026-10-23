"""Check that planted conflicts and the expired policy are reachable by plain BM25 (offline).

For each entry in data/eval/kb_conflicts.json the probe query (typed without diacritics, as customers often do)
is ranked against all chunks; the script prints the rank of each side. A conflict that BM25 cannot surface
would only test retrieval, not conflict handling, so both sides should appear near the top.

    python scripts/probe_conflicts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.bm25 import BM25  # noqa: E402
from rag_cskh.chunking import chunk_corpus  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.text import tokens  # noqa: E402


def best_rank(ranked: list[str], citations: list[str]) -> int | None:
    ranks = [ranked.index(c) + 1 for c in citations if c in ranked]
    return min(ranks) if ranks else None


def probe() -> list[dict]:
    registry = json.loads((ROOT / "data" / "eval" / "kb_conflicts.json").read_text(encoding="utf-8"))
    chunks = chunk_corpus(load_corpus(), 120)
    bm25 = BM25([tokens(c.text) for c in chunks])
    rows = []
    for entry in registry["conflicts"] + registry["superseded"]:
        ranked = list(dict.fromkeys(chunks[i].citation for i in bm25.rank(tokens(entry["probe_query"]))))
        side_a, side_b = entry["sections"][0], [entry["sections"][1]] + entry.get("also_conflicts_with", [])
        rows.append({"id": entry["id"], "query": entry["probe_query"], "rank_a": best_rank(ranked, [side_a]),
                     "rank_b": best_rank(ranked, side_b), "top1": ranked[0] if ranked else None})
    return rows


def main() -> None:
    print("| Id | Probe query | Rank side A | Rank side B | Top-1 section |")
    print("|---|---|---:|---:|---|")
    for r in probe():
        print(f"| {r['id']} | {r['query']} | {r['rank_a']} | {r['rank_b']} | {r['top1']} |")


if __name__ == "__main__":
    main()
