"""Week-6 baselines end to end: Vector RAG and Hybrid RAG (+ Hybrid RAG without reranker as an ablation).

    python scripts/run_baselines.py --limit 3            # trial on 3 dev tickets
    python scripts/run_baselines.py                      # dev split, all 51 tickets (~USD 0.03 with gpt-4o-mini)
    python scripts/run_baselines.py --report-only        # rebuild the report from the cache, no API calls
    python scripts/run_baselines.py --ce bge-m3          # + Hybrid RAG with the cross-encoder reranker (local model)

All four ticket types are run (no_evidence included). The test split is kept for the final comparison in Week 10.
Answer model: LLM_MODEL; reranker: RERANK_MODEL or LLM_MODEL (reranks from eval_rerank.py are reused from cache).
Outputs: reports/week6/answers_<split>.jsonl (one row per system and ticket) and baselines_<split>.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.answer import DECISIONS  # noqa: E402
from rag_cskh.answer_metrics import confusion, macro_f1, mean_of, ticket_answer_metrics  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.embeddings import load_env, openai_embedder  # noqa: E402
from rag_cskh.llm import openai_llm  # noqa: E402
from rag_cskh.pipelines import RAGPipeline, build_baselines  # noqa: E402
from rag_cskh.tickets import load_tickets  # noqa: E402

OUT = ROOT / "reports" / "week6"
TYPES = ("normal", "missing_info", "no_evidence", "conflict")
SHORT = {"normal": "N", "missing_info": "M", "no_evidence": "E", "conflict": "K"}


def fmt(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def run_system(pipe: RAGPipeline, tickets: list[dict], workers: int) -> list[dict]:
    def one(t):
        out = pipe.run(t["ticket"])
        return {"system": pipe.name, "id": t["id"], "type": t["type"], "expected": t["expected_decision"],
                "evidence": out["evidence"], "first_stage": out["first_stage"][:10], "grades": out["grades"],
                "answer": out["answer"], "metrics": ticket_answer_metrics(t, out["evidence"], out["answer"])}
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, tickets))


def system_row(name: str, rows: list[dict]) -> str:
    ms = [r["metrics"] for r in rows]
    by = {k: [r["metrics"] for r in rows if r["type"] == k] for k in TYPES}
    acc = {k: mean_of(v, "decision_ok") for k, v in by.items()}
    f1 = macro_f1([r["expected"] for r in rows], [r["answer"]["decision"] for r in rows])
    ctx_n, ctx_k = mean_of(by["normal"], "context_ok"), mean_of(by["conflict"], "context_ok")
    return (f"| {name} | {fmt(mean_of(ms, 'decision_ok'))} | {fmt(f1)} | " + " | ".join(fmt(acc[k]) for k in TYPES) +
            f" | {fmt(ctx_n)} | {fmt(ctx_k)} | {fmt(mean_of(by['normal'], 'fact_cov'))} | {fmt(mean_of(by['missing_info'], 'slot_cov'))} | "
            f"{fmt(mean_of(by['normal'], 'cited_ok'))} | {fmt(mean_of(by['conflict'], 'cited_ok'))} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-ablation", action="store_true", help="skip Hybrid RAG without reranker")
    parser.add_argument("--ce", default="", help="also run hybrid-rag-ce with this cross-encoder preset (bge-m3, phoranker, ...)")
    parser.add_argument("--device", default=None, help="cross-encoder device: cpu | cuda (default: auto)")
    parser.add_argument("--report-only", action="store_true", help="use cached calls only")
    args = parser.parse_args()

    load_env()
    tickets = [t for t in load_tickets() if args.split == "all" or t["split"] == args.split]
    if args.limit:
        tickets = tickets[:args.limit]
    docs = load_corpus()
    llm = openai_llm()
    rerank_llm = openai_llm(os.getenv("RERANK_MODEL")) if os.getenv("RERANK_MODEL") else llm
    if args.report_only:
        def no_api(*_):
            raise RuntimeError("uncached call in --report-only mode")
        llm.raw_call = rerank_llm.raw_call = no_api
    systems = build_baselines(docs, openai_embedder(), llm, rerank_llm)
    if not args.no_ablation:
        h = systems["hybrid-rag"]
        systems["hybrid-rag-norerank"] = RAGPipeline("hybrid-rag-norerank", h.retriever, h.store, llm)
    if args.ce:
        from rag_cskh.cross_encoder import load_cross_encoder
        h = systems["hybrid-rag"]
        ce = load_cross_encoder(h.store, args.ce, args.device)
        systems["hybrid-rag-ce"] = RAGPipeline("hybrid-rag-ce", h.retriever, h.store, llm, ce)

    rows: dict[str, list[dict]] = {}
    for name, pipe in systems.items():
        local_model = pipe.reranker is not None and pipe.reranker.name.startswith("ce:")
        rows[name] = run_system(pipe, tickets, 1 if local_model else args.workers)   # one torch model: no threads
        print(f"{name}: {len(rows[name])} tickets done")

    counts = " ".join(f"{SHORT[k]}={sum(t['type'] == k for t in tickets)}" for k in TYPES)
    lines = [f"# Week-6 baselines ({args.split} split, {len(tickets)} tickets: {counts}; answer model {llm.model})\n",
             "Decision accuracy by expected type; context = gold evidence among the 5 sections given to the model; "
             "fact/slot coverage are lexical approximations (see answer_metrics.py); cited = gold section(s) in the citations.\n",
             "| System | Decision acc | Macro-F1 | N acc | M acc | E acc | K acc | N context | K context (both sides) | N fact cov | M slot cov | N cited | K both cited |",
             "|---|" + "---:|" * 12]
    for name, r in rows.items():
        lines.append(system_row(name, r))

    lines.append("\n## Format, citations and cost\n")
    lines.append("| System | Words (mean) | Over 120 words | Code shown to customer | Invalid citations | Tokens / ticket (answer + rerank) | Answer latency / ticket (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, r in rows.items():
        ms = [x["metrics"] for x in r]
        ans_tokens = mean(x["answer"]["tokens"] for x in r)
        lines.append(f"| {name} | {mean(m['words'] for m in ms):.0f} | {fmt(mean_of(ms, 'over_120'))} | {fmt(mean_of(ms, 'code_leak'))} | "
                     f"{sum(m['invalid_cit'] for m in ms)} | {ans_tokens:,.0f}{' + rerank' if systems[name].reranker else ''} | "
                     f"{mean(x['answer']['latency_s'] for x in r):.1f} |")
    lines.append(f"\nAll calls: answer model {llm.model} {llm.usage.prompt_tokens:,} input + {llm.usage.completion_tokens:,} output tokens "
                 f"(estimated USD {llm.usage.cost(llm.model):.3f}); reranker calls are counted in rerank_<split>.md when made there.")

    lines.append("\n## Decision confusion (rows = expected, columns = predicted)\n")
    for name, r in rows.items():
        cm = confusion([x["expected"] for x in r], [x["answer"]["decision"] for x in r])
        lines.append(f"**{name}**\n\n| | " + " | ".join(DECISIONS) + " |\n|---|" + "---:|" * len(DECISIONS))
        for g in DECISIONS:
            lines.append(f"| {g} | " + " | ".join(str(cm[(g, p)]) for p in DECISIONS) + " |")
        lines.append("")

    lines.append("## Tickets with a wrong decision in at least one system\n")
    names = list(rows)
    lines.append("| Id | Type | Expected | " + " | ".join(names) + " | Ticket |\n|---|---|---|" + "---|" * len(names) + "---|")
    by_id = {t["id"]: t for t in tickets}
    index = {n: {x["id"]: x for x in rows[n]} for n in names}
    for t in tickets:
        preds = [index[n][t["id"]]["answer"]["decision"] for n in names]
        if any(p != t["expected_decision"] for p in preds):
            cells = " | ".join(p + ("" if index[n][t["id"]]["metrics"]["context_ok"] in (None, 1.0) else " (ctx miss)")
                               for n, p in zip(names, preds))
            lines.append(f"| {t['id']} | {t['type']} | {t['expected_decision']} | {cells} | {by_id[t['id']]['ticket'][:80].replace('|', '/')} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"baselines_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / f"answers_{args.split}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows.values():
            for x in r:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
