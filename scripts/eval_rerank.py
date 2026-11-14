"""Week-6 reranker evaluation: hybrid RRF top 20 vs LLM reranker vs cross-encoder, section-level retrieval metrics.

    python scripts/eval_rerank.py --split dev --limit 3                 # trial (3 tickets)
    python scripts/eval_rerank.py --split dev                           # LLM reranker (~USD 0.03 with gpt-4o-mini)
    python scripts/eval_rerank.py --split dev --rerankers llm,ce        # + cross-encoder (local, bge-reranker-v2-m3)
    python scripts/eval_rerank.py --split test --rerankers llm,ce --ce-model phoranker
    python scripts/eval_rerank.py --split dev --rerankers llm,ce --report-only   # no API calls (LLM from cache)

Methods: hybrid (first stage), hybrid+llm (LLMReranker), hybrid+ce (CrossEncoderReranker) and hybrid+ce-fused
(RRF of the hybrid ranking and the cross-encoder ranking, a common safeguard when a cross-encoder is not tuned
on the domain). Same tickets and metrics as Week 5 (normal, conflict, missing_info). LLM model: RERANK_MODEL, else
LLM_MODEL. LLM calls and cross-encoder scores are cached in data/index/, so the baselines reuse them.
Outputs: reports/week6/rerank_<split>[_<ce-model>].md and rerank_per_ticket_<split>[_<ce-model>].csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.chunking import chunk_corpus  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.embeddings import load_env, openai_embedder  # noqa: E402
from rag_cskh.evidence import SectionStore  # noqa: E402
from rag_cskh.llm import openai_llm  # noqa: E402
from rag_cskh.metrics import average, bootstrap_ci, paired_diff_ci, ticket_metrics  # noqa: E402
from rag_cskh.rerank import RERANK_DEPTH, LLMReranker  # noqa: E402
from rag_cskh.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, rrf  # noqa: E402
from rag_cskh.tickets import SNAPSHOT, load_tickets  # noqa: E402

TYPES = ("normal", "conflict", "missing_info")
TAGS = ("no_diacritics", "multi_doc", "catalog", "manufacturer", "near_miss")
OUT = ROOT / "reports" / "week6"


def fmt(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def rerank_all(reranker, tickets, first, workers: int) -> dict[str, tuple[list[str], dict]]:
    with ThreadPoolExecutor(workers) as pool:
        results = pool.map(lambda t: reranker.rerank(t["ticket"], first[t["id"]]), tickets)
        return {t["id"]: r for t, r in zip(tickets, results)}


def fuse(first: list[str], reranked: list[str]) -> list[str]:
    ids = {c: i for i, c in enumerate(first)}
    return [first[i] for i in rrf([[ids[c] for c in first], [ids[c] for c in reranked]])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--rerankers", default="llm", help="comma-separated: llm, ce")
    parser.add_argument("--ce-model", default="bge-m3", help="bge-m3 | phoranker | viranker | Hugging Face id")
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    parser.add_argument("--fp16", action="store_true", help="half precision on GPU")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true", help="no API calls: LLM grades from cache only")
    args = parser.parse_args()
    use = [r.strip() for r in args.rerankers.split(",") if r.strip()]

    load_env()
    tickets = [t for t in load_tickets() if t["type"] in TYPES and (args.split == "all" or t["split"] == args.split)]
    if args.limit:
        tickets = tickets[:args.limit]
    docs = load_corpus()
    store = SectionStore(docs)
    chunks = chunk_corpus(docs, 120)
    hybrid = HybridRetriever([BM25Retriever(chunks, active_on=SNAPSHOT), DenseRetriever(chunks, openai_embedder(), SNAPSHOT)])
    first = {t["id"]: hybrid.search_sections(t["ticket"], RERANK_DEPTH) for t in tickets}
    runs: dict[str, dict[str, list[str]]] = {"hybrid": first}
    extra: dict[str, dict] = {}
    cost_lines = []

    if "llm" in use:
        llm = openai_llm(os.getenv("RERANK_MODEL") or None)
        if args.report_only:
            def no_api(*_):
                raise RuntimeError("uncached call in --report-only mode")
            llm.raw_call = no_api
        out = rerank_all(LLMReranker(llm, store), tickets, first, args.workers)
        runs["hybrid+llm"] = {i: r[0] for i, r in out.items()}
        extra["llm"] = {i: r[1] for i, r in out.items()}
        u = llm.usage
        cost_lines.append(f"- LLM reranker ({llm.model}): {u.calls} calls ({u.cached} from cache), {u.prompt_tokens / max(u.calls, 1):,.0f} "
                          f"input + {u.completion_tokens / max(u.calls, 1):,.0f} output tokens and {u.latency_s / max(u.calls, 1):.1f} s per "
                          f"ticket (original calls), estimated USD {u.cost(llm.model):.3f} in total.")
    if "ce" in use:
        from rag_cskh.cross_encoder import load_cross_encoder
        start = time.perf_counter()
        ce = load_cross_encoder(store, args.ce_model, args.device, args.fp16)
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        out = {t["id"]: ce.rerank(t["ticket"], first[t["id"]]) for t in tickets}   # one model, sequential batches
        run_s = time.perf_counter() - start
        runs["hybrid+ce"] = {i: r[0] for i, r in out.items()}
        runs["hybrid+ce-fused"] = {i: fuse(first[i], r[0]) for i, r in out.items()}
        extra["ce"] = {i: r[1] for i, r in out.items()}
        cost_lines.append(f"- Cross-encoder {ce.model_name}: {ce.pairs_scored} pairs scored now ({len(tickets) * RERANK_DEPTH} in total, "
                          f"rest from cache), model load {load_s:.0f} s, {run_s / max(len(tickets), 1):.2f} s per ticket, no API cost.")

    per = {m: {t["id"]: ticket_metrics(runs[m][t["id"]], t["gold_sections"], t["type"] == "conflict") for t in tickets}
           for m in runs}
    title = " + ".join(["hybrid"] + [m for m in runs if m != "hybrid"])
    lines = [f"# Week-6 reranker results ({args.split} split, {len(tickets)} tickets, depth {RERANK_DEPTH}"
             + (f", cross-encoder {args.ce_model}" if "ce" in use else "") + f")\n\nMethods: {title}\n",
             "## Main results (citation level)\n",
             "| Method | N: Hit@1 | N: Recall@3 | N: Recall@5 | N: All@5 | N: MRR | N: nDCG@5 | K: Both@3 | K: Both@5 | M: Hit@1 | M: Hit@5 |",
             "|---|" + "---:|" * 10]
    for m in runs:
        rows = {k: average([per[m][t["id"]] for t in tickets if t["type"] == k]) for k in TYPES
                if any(t["type"] == k for t in tickets)}
        n, k, mi = rows.get("normal", {}), rows.get("conflict", {}), rows.get("missing_info", {})
        lines.append(f"| {m} | {fmt(n.get('Hit@1'))} | {fmt(n.get('Recall@3'))} | {fmt(n.get('Recall@5'))} | {fmt(n.get('All@5'))} | "
                     f"{fmt(n.get('MRR'))} | {fmt(n.get('nDCG@5'))} | {fmt(k.get('Both@3'))} | {fmt(k.get('Both@5'))} | "
                     f"{fmt(mi.get('Hit@1'))} | {fmt(mi.get('Hit@5'))} |")

    scored = [t for t in tickets if t["type"] in ("normal", "conflict")]
    pairs = [("hybrid", m) for m in runs if m != "hybrid"]
    if "hybrid+llm" in runs and "hybrid+ce" in runs:
        pairs += [("hybrid+llm", "hybrid+ce"), ("hybrid+llm", "hybrid+ce-fused")]
    if pairs:
        lines.append(f"\n## Paired differences (B - A, normal + conflict, n={len(scored)}, 95% bootstrap CI)\n")
        lines.append("| A | B | diff MRR | diff Hit@1 | diff Recall@5 | diff nDCG@5 |\n|---|---|---|---|---|---|")
        for a, b in pairs:
            cells = []
            for metric in ("MRR", "Hit@1", "Recall@5", "nDCG@5"):
                d, lo, hi = paired_diff_ci([per[a][t["id"]][metric] for t in scored], [per[b][t["id"]][metric] for t in scored])
                cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {a} | {b} | " + " | ".join(cells) + " |")
        lines.append("\nMRR 95% CI: " + "; ".join(
            f"{m} [{fmt(bootstrap_ci([per[m][t['id']]['MRR'] for t in scored])[0])}, "
            f"{fmt(bootstrap_ci([per[m][t['id']]['MRR'] for t in scored])[1])}]" for m in runs))

    conflicts = [t for t in tickets if t["type"] == "conflict"]
    if conflicts:
        lines.append("\n## Conflict tickets by registered conflict (Both@5: both sides in the top 5)\n")
        lines.append("| Conflict | n | " + " | ".join(runs) + " |\n|---|---:|" + "---:|" * len(runs))
        by_c = defaultdict(list)
        for t in conflicts:
            by_c[t["conflict_id"]].append(t)
        for cid in sorted(by_c):
            g = by_c[cid]
            lines.append(f"| {cid} | {len(g)} | " + " | ".join(fmt(mean(per[m][t['id']]['Both@5'] for t in g)) for m in runs) + " |")

    lines.append("\n## By tag (Recall@5, normal + conflict tickets with the tag)\n")
    lines.append("| Tag | n | " + " | ".join(runs) + " |\n|---|---:|" + "---:|" * len(runs))
    for tag in TAGS:
        group = [t for t in scored if tag in t["tags"]]
        if group:
            lines.append(f"| {tag} | {len(group)} | " + " | ".join(
                fmt(mean(per[m][t['id']]['Recall@5'] for t in group)) for m in runs) + " |")

    lines.append("\n## Scores given to gold and other candidates\n")
    for kind, scores in extra.items():
        gold_s, other_s, low = [], [], []
        for t in tickets:
            for c, s in scores[t["id"]].items():
                (gold_s if c in t["gold_sections"] else other_s).append(s)
        if kind == "llm":
            for t in tickets:
                low += [f"{t['id']}:{c}={g}" for c, g in extra["llm"][t["id"]].items() if c in t["gold_sections"] and g <= 1]
            lines.append(f"- LLM grade (0-3): gold {fmt(mean(gold_s) if gold_s else None)} (n={len(gold_s)}), "
                         f"other {fmt(mean(other_s) if other_s else None)} (n={len(other_s)}); ungraded {sum(s < 0 for s in gold_s + other_s)}; "
                         f"gold graded 0-1 ({len(low)}): {', '.join(low) or 'none'}")
        else:
            ranks = [sorted(scores[t["id"]].values(), reverse=True).index(scores[t["id"]][c]) + 1
                     for t in tickets for c in t["gold_sections"] if c in scores[t["id"]]]
            lines.append(f"- Cross-encoder score: gold {fmt(mean(gold_s) if gold_s else None)}, other "
                         f"{fmt(mean(other_s) if other_s else None)}; mean rank of gold sections among the 20 candidates "
                         f"{fmt(mean(ranks) if ranks else None)}")
    for m in runs:
        if m == "hybrid":
            continue
        won = [t["id"] for t in scored if per["hybrid"][t["id"]]["Recall@5"] < per[m][t["id"]]["Recall@5"]]
        lost = [t["id"] for t in scored if per["hybrid"][t["id"]]["Recall@5"] > per[m][t["id"]]["Recall@5"]]
        lines.append(f"- {m}: Recall@5 better than hybrid on {len(won)} tickets ({' '.join(won) or '-'}), "
                     f"worse on {len(lost)} ({' '.join(lost) or '-'})")

    if cost_lines:
        lines.append("\n## Cost\n")
        lines += cost_lines

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = args.split + (f"_{args.ce_model.replace('/', '_')}" if "ce" in use else "")
    (OUT / f"rerank_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / f"rerank_per_ticket_{suffix}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        methods = [m for m in runs if m != "hybrid"]
        w.writerow(["id", "type", "conflict_id", "tags", "gold", "hybrid_top5"] + [f"{m}_top5" for m in methods]
                   + [f"MRR_{m}" for m in runs] + [f"Recall@5_{m}" for m in runs])
        for t in tickets:
            w.writerow([t["id"], t["type"], t.get("conflict_id", ""), " ".join(t["tags"]), " ".join(t["gold_sections"]),
                        " ".join(first[t["id"]][:5])] + [" ".join(runs[m][t["id"]][:5]) for m in methods]
                       + [fmt(per[m][t["id"]]["MRR"]) for m in runs] + [fmt(per[m][t["id"]]["Recall@5"]) for m in runs])
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
