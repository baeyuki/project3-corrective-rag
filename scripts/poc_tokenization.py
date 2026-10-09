"""Feasibility check: how much does Vietnamese-aware tokenisation help BM25?

Ranks a toy corpus (data/poc/toy_kb.jsonl) for 12 queries, half typed without diacritics, with three tokenisers:
  words    : lower-cased whitespace tokens
  syllable : syllables + bigrams
  folded   : syllables + bigrams + accent-folded copies (the project's tokeniser)

    python scripts/poc_tokenization.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.bm25 import BM25  # noqa: E402
from rag_cskh.text import tokens  # noqa: E402

TOKENISERS = {
    "words": lambda t: re.findall(r"\w+", t.lower()),
    "syllable": lambda t: tokens(t, bigrams=True, folded=False),
    "folded": lambda t: tokens(t, bigrams=True, folded=True),
}


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "poc" / name).read_text(encoding="utf-8").splitlines() if line]


def evaluate(tok, docs, queries) -> dict:
    bm25 = BM25([tok(d["text"]) for d in docs])
    out = {}
    for accented in (True, False):
        qs = [q for q in queries if (q["query"] != q["query"].encode("ascii", "ignore").decode()) == accented]
        hits, rr = 0, 0.0
        for q in qs:
            ranked = [docs[i]["id"] for i in bm25.rank(tok(q["query"]))]
            if ranked[:1] == [q["gold"]]:
                hits += 1
            if q["gold"] in ranked:
                rr += 1 / (ranked.index(q["gold"]) + 1)
        out["with diacritics" if accented else "no diacritics"] = (hits / len(qs), rr / len(qs))
    return out


def main() -> None:
    docs, queries = load("toy_kb.jsonl"), load("toy_queries.jsonl")
    print(f"{len(docs)} passages, {len(queries)} queries\n")
    print(f"{'tokeniser':10s} {'with diacritics (Hit@1 / MRR)':32s} no diacritics (Hit@1 / MRR)")
    for name, tok in TOKENISERS.items():
        r = evaluate(tok, docs, queries)
        a, b = r["with diacritics"], r["no diacritics"]
        print(f"{name:10s} {a[0]:.2f} / {a[1]:.2f}{'':21s} {b[0]:.2f} / {b[1]:.2f}")


if __name__ == "__main__":
    main()
