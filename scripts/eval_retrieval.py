"""Week-5 retrieval evaluation: BM25, dense (OpenAI embeddings), hybrid RRF.

    python scripts/eval_retrieval.py                          # BM25 variants only (offline)
    python scripts/eval_retrieval.py --dense                  # + vector and hybrid (needs OPENAI_API_KEY, < $0.01)
    python scripts/eval_retrieval.py --dense --split test     # final numbers on the test split

Queries are the raw ticket texts (no rewriting: that is Corrective RAG, Weeks 7-8). Evaluated tickets: normal,
conflict and missing_info (the latter against the section that states the required information); no_evidence
tickets have no gold sections. Results go to reports/week5/ (git-ignored): summary.md, results.json and
per_ticket.csv for error analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.chunking import chunk_corpus  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.metrics import KS, average, bootstrap_ci, paired_diff_ci, ticket_metrics  # noqa: E402
from rag_cskh.retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from rag_cskh.text import tokens  # noqa: E402
from rag_cskh.tickets import SNAPSHOT, load_tickets  # noqa: E402

GRANULARITIES = {"section": None, "max-120": 120, "max-60": 60}
TYPES = ("normal", "conflict", "missing_info")
TAGS = ("no_diacritics", "multi_doc", "catalog", "manufacturer", "near_miss", "expired_trap")
DEPTH = 20
OUT = ROOT / "reports" / "week5"

TOKENIZERS = {
    "bm25-words": lambda t: re.findall(r"\w+", t.lower()),
    "bm25-syllable": lambda t: tokens(t, bigrams=True, folded=False),
    "bm25": tokens,
}


def build_methods(chunks, dense: bool, active_on):
    methods = {name: BM25Retriever(chunks, tok, active_on, name) for name, tok in TOKENIZERS.items()}
    if dense:
        from rag_cskh.embeddings import openai_embedder
        embedder = openai_embedder()
        methods["vector"] = DenseRetriever(chunks, embedder, active_on)
        methods["hybrid"] = HybridRetriever([methods["bm25"], methods["vector"]])
    return methods


def evaluate(tickets, methods) -> dict[str, dict[str, dict]]:
    """per_ticket[method][ticket_id] = {"ranked": [...], "metrics": {...}}"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for name, retriever in methods.items():
        for t in tickets:
            ranked = retriever.search_sections(t["ticket"], DEPTH)
            out[name][t["id"]] = {"ranked": ranked,
                                  "metrics": ticket_metrics(ranked, t["gold_sections"], t["type"] == "conflict")}
    return out


def fmt(x: float) -> str:
    return f"{x:.2f}"


def summary_rows(tickets, per_ticket, method):
    rows = {}
    for kind in TYPES:
        ms = [per_ticket[method][t["id"]]["metrics"] for t in tickets if t["type"] == kind]
        rows[kind] = average(ms) if ms else {}
    return rows


def expired_rate(tickets, chunks, k=5) -> float:
    """Share of tickets whose unfiltered BM25 top-k sections include an expired document."""
    r = BM25Retriever(chunks)
    expired = {c.citation for c in chunks if c.status == "expired"}
    return sum(bool(set(r.search_sections(t["ticket"], k)) & expired) for t in tickets) / len(tickets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true", help="add vector and hybrid (OpenAI embeddings, cached)")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--no-filter", action="store_true", help="do not drop expired documents")
    args = parser.parse_args()

    tickets = [t for t in load_tickets() if t["type"] in TYPES and (args.split == "all" or t["split"] == args.split)]
    docs = load_corpus()
    active_on = None if args.no_filter else SNAPSHOT
    lines = [f"# Week-5 retrieval results ({args.split} split, {len(tickets)} tickets, "
             f"validity filter {'off' if args.no_filter else 'on'})\n"]
    results, all_per_ticket = {}, {}

    for gran, max_words in GRANULARITIES.items():
        chunks = chunk_corpus(docs, max_words)
        methods = build_methods(chunks, args.dense, active_on)
        per_ticket = evaluate(tickets, methods)
        all_per_ticket[gran] = per_ticket
        results[gran] = {m: summary_rows(tickets, per_ticket, m) for m in methods}

    # main table
    lines.append("## Main results (citation level)\n")
    lines.append("| Granularity | Method | N: Hit@1 | N: Recall@5 | N: All@5 | N: MRR | N: nDCG@10 | K: Both@5 | K: Both@10 | M: Hit@5 | M: MRR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for gran, by_method in results.items():
        for m, rows in by_method.items():
            n, k, mi = rows["normal"], rows["conflict"], rows["missing_info"]
            lines.append(f"| {gran} | {m} | {fmt(n['Hit@1'])} | {fmt(n['Recall@5'])} | {fmt(n['All@5'])} | {fmt(n['MRR'])} | "
                         f"{fmt(n['nDCG@10'])} | {fmt(k['Both@5'])} | {fmt(k['Both@10'])} | {fmt(mi['Hit@5'])} | {fmt(mi['MRR'])} |")

    # recall curve and CIs for the default granularity
    gran = "max-120"
    per_ticket = all_per_ticket[gran]
    scored = [t for t in tickets if t["type"] in ("normal", "conflict")]
    lines.append(f"\n## Recall@K, normal + conflict tickets ({gran}, n={len(scored)})\n")
    lines.append("| Method | " + " | ".join(f"Recall@{k}" for k in KS) + " | MRR [95% CI] |")
    lines.append("|---|" + "---:|" * (len(KS) + 1))
    for m in per_ticket:
        vals = [per_ticket[m][t["id"]]["metrics"] for t in scored]
        rec = " | ".join(fmt(sum(v[f"Recall@{k}"] for v in vals) / len(vals)) for k in KS)
        mrr = [v["MRR"] for v in vals]
        lo, hi = bootstrap_ci(mrr)
        lines.append(f"| {m} | {rec} | {fmt(sum(mrr) / len(mrr))} [{fmt(lo)}, {fmt(hi)}] |")

    if "hybrid" in per_ticket:
        lines.append(f"\n## Paired differences vs BM25 ({gran}, normal + conflict, 95% bootstrap CI)\n")
        lines.append("| Method | diff MRR | diff Recall@5 |\n|---|---|---|")
        base = per_ticket["bm25"]
        for m in ("vector", "hybrid"):
            out = []
            for metric in ("MRR", "Recall@5"):
                d, lo, hi = paired_diff_ci([base[t["id"]]["metrics"][metric] for t in scored],
                                           [per_ticket[m][t["id"]]["metrics"][metric] for t in scored])
                out.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {m} | {out[0]} | {out[1]} |")

    lines.append(f"\n## By tag ({gran}; Recall@5 on normal + conflict tickets with the tag)\n")
    lines.append("| Tag | n | " + " | ".join(per_ticket) + " |\n|---|---:|" + "---:|" * len(per_ticket))
    for tag in TAGS:
        group = [t for t in scored if tag in t["tags"]]
        if group:
            cells = " | ".join(fmt(sum(per_ticket[m][t["id"]]["metrics"]["Recall@5"] for t in group) / len(group)) for m in per_ticket)
            lines.append(f"| {tag} | {len(group)} | {cells} |")

    if not args.no_filter:
        rate = expired_rate(tickets, chunk_corpus(docs, 120))
        lines.append(f"\nWithout the validity filter, BM25 puts the expired policy in the top 5 sections for {rate:.0%} of these tickets.")

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}{'_nofilter' if args.no_filter else ''}"
    (OUT / f"summary_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"results_{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (OUT / f"per_ticket_{suffix}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["granularity", "method", "id", "type", "tags", "gold", "top5", "MRR", "Recall@5"])
        by_id = {t["id"]: t for t in tickets}
        for g, pt in all_per_ticket.items():
            for m, rows in pt.items():
                for tid, row in rows.items():
                    t = by_id[tid]
                    w.writerow([g, m, tid, t["type"], " ".join(t["tags"]), " ".join(t["gold_sections"]),
                                " ".join(row["ranked"][:5]), fmt(row["metrics"]["MRR"]), fmt(row["metrics"]["Recall@5"])])
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
