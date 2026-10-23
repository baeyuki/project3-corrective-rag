"""Parse the knowledge base, chunk it and print corpus statistics.

    python scripts/build_corpus.py                       # default: sections split at 120 words
    python scripts/build_corpus.py --max-words 60        # finer chunks
    python scripts/build_corpus.py --on 2026-10-22       # date used for the active/expired split

Writes data/index/chunks.jsonl (one chunk per line) and data/index/corpus_stats.json; data/index/ is git-ignored
because both files are fully reproducible from data/kb/.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.chunking import DEFAULT_MAX_WORDS, chunk_corpus, n_words  # noqa: E402
from rag_cskh.corpus import CATEGORIES, load_corpus  # noqa: E402

SNAPSHOT = "2026-10-22"
GRANULARITIES = {"section": None, "max-120": 120, "max-60": 60}


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def length_summary(values: list[int]) -> dict:
    return {
        "min": min(values),
        "median": int(statistics.median(values)),
        "p90": percentile(values, 0.9),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
    }


def build_stats(docs, chunks, on: date) -> dict:
    by_cat = defaultdict(lambda: {"documents": 0, "sections": 0, "chunks": 0, "words": 0})
    for d in docs:
        row = by_cat[d.category]
        row["documents"] += 1
        row["sections"] += len(d.sections)
        row["words"] += sum(n_words(s.body) for s in d.sections)
    for c in chunks:
        by_cat[c.category]["chunks"] += 1
    sections = [s for d in docs for s in d.sections]
    granularity = {}
    for name, max_words in GRANULARITIES.items():
        cs = chunk_corpus(docs, max_words)
        granularity[name] = {"chunks": len(cs), "words": length_summary([c.n_words for c in cs])}
    return {
        "snapshot": on.isoformat(),
        "documents": len(docs),
        "sections": len(sections),
        "chunks": len(chunks),
        "words": sum(n_words(s.body) for s in sections),
        "characters": sum(len(s.body) for s in sections),
        "by_category": {cat: by_cat[cat] for cat in CATEGORIES if cat in by_cat},
        "status": dict(Counter(d.meta["status"] for d in docs)),
        "usable_on_snapshot": sum(d.is_active(on) for d in docs),
        "provenance": dict(Counter(d.meta["provenance"] for d in docs)),
        "brands": dict(Counter(b.strip() for d in docs for b in d.meta.get("brand", "").split(",") if b.strip())),
        "section_words": length_summary([n_words(s.body) for s in sections]),
        "chunk_words": length_summary([c.n_words for c in chunks]),
        "granularity": granularity,
    }


def print_markdown(stats: dict) -> None:
    print(f"Snapshot {stats['snapshot']}: {stats['documents']} documents, {stats['sections']} sections, "
          f"{stats['chunks']} chunks, {stats['words']:,} words, {stats['characters']:,} characters\n")
    print("| Category | Documents | Sections | Chunks | Words |")
    print("|---|---:|---:|---:|---:|")
    for cat, row in stats["by_category"].items():
        print(f"| {cat} | {row['documents']} | {row['sections']} | {row['chunks']} | {row['words']:,} |")
    print(f"| **Total** | **{stats['documents']}** | **{stats['sections']}** | **{stats['chunks']}** | **{stats['words']:,}** |\n")
    print("| Granularity | Chunks | Words min / median / p90 / max |")
    print("|---|---:|---|")
    for name, g in stats["granularity"].items():
        w = g["words"]
        print(f"| {name} | {g['chunks']} | {w['min']} / {w['median']} / {w['p90']} / {w['max']} |")
    print(f"\nStatus: {stats['status']}; usable on snapshot: {stats['usable_on_snapshot']}")
    print(f"Provenance: {stats['provenance']}")
    print(f"Brands: {stats['brands']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="0 = one chunk per section")
    parser.add_argument("--on", default=SNAPSHOT, help="date for the active/expired check (YYYY-MM-DD)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "index")
    args = parser.parse_args()

    docs = load_corpus()
    chunks = chunk_corpus(docs, args.max_words or None)
    stats = build_stats(docs, chunks, date.fromisoformat(args.on))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (args.out / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print_markdown(stats)
    print(f"\nWrote {len(chunks)} chunks to {args.out / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
