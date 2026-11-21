"""Week-7 evaluation of Corrective RAG part I: ticket analysis, multi-query retrieval, evidence grader.

    python scripts/eval_grader.py --limit 3                 # trial on 3 dev tickets
    python scripts/eval_grader.py                           # dev split, 52 tickets (~USD 0.10 with gpt-4o-mini)
    python scripts/eval_grader.py --grader-model gpt-4.1-mini   # same pipeline, stronger grader
    python scripts/eval_grader.py --report-only             # rebuild from the cache, no API calls
v1 results (before the dev iteration) are kept in reports/week7/grader_dev_v1.md.

Systems: crag-multi (analysis -> multi-query hybrid -> LLM reranker -> grader on the top 10) and crag-single
(same without multi-query: raw ticket only, i.e. the Week-6 Hybrid RAG candidates). The decision comes from the
grader (answer / clarify / escalate); no reply is generated yet (Week 8). Reference: Week-6 Hybrid RAG decisions
(reports/week6/answers_dev.jsonl). Also runs the registry probes of the six planted conflicts.
Models: LLM_MODEL for analysis and reranking, GRADER_MODEL (else LLM_MODEL) for grading.
Outputs: reports/week7/grader_<split>[_<model>].md and per_ticket_<split>[_<model>].jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.analysis import analyze  # noqa: E402
from rag_cskh.answer import DECISIONS  # noqa: E402
from rag_cskh.answer_metrics import confusion, covered, macro_f1  # noqa: E402
from rag_cskh.chunking import chunk_corpus  # noqa: E402
from rag_cskh.corpus import load_corpus  # noqa: E402
from rag_cskh.crag import CorrectiveRetriever  # noqa: E402
from rag_cskh.embeddings import load_env, openai_embedder  # noqa: E402
from rag_cskh.evidence import SectionStore  # noqa: E402
from rag_cskh.llm import openai_llm  # noqa: E402
from rag_cskh.metrics import ticket_metrics  # noqa: E402
from rag_cskh.retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from rag_cskh.tickets import SNAPSHOT, load_conflicts, load_tickets  # noqa: E402

OUT = ROOT / "reports" / "week7"
TYPES = ("normal", "missing_info", "no_evidence", "conflict")
SHORT = {"normal": "N", "missing_info": "M", "no_evidence": "E", "conflict": "K"}
EXPECTED_TRIGGER = {"normal": "sufficient", "missing_info": "missing_info", "no_evidence": "no_evidence",
                    "conflict": "conflict"}


def fmt(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def avg(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def conflict_hit(t: dict, conflicts: list[dict]) -> bool:
    """A valid conflict whose two sides are side A and a side B of the gold pair."""
    a, bs = t["gold_sections"][0], set(t["gold_sections"][1:])
    return any({x["a"], x["b"]} & {a} and {x["a"], x["b"]} & bs for x in conflicts)


def per_ticket(t: dict, out: dict) -> dict:
    g = out["grade"]
    gold = t["gold_sections"]
    m = {"decision_ok": float(g["decision"] == t["expected_decision"]),
         "trigger_ok": float(g["trigger"] == EXPECTED_TRIGGER[t["type"]]),
         "false_conflict": float(t["type"] != "conflict" and bool(g["conflicts"]))}
    if gold:
        tm = ticket_metrics(out["ranked"], gold, t["type"] == "conflict")
        m.update({"Recall@5": tm["Recall@5"], "Recall@10": tm["Recall@10"], "Hit@10": tm["Hit@10"],
                  "Both@10": tm.get("Both@10"), "Recall@20_fused": ticket_metrics(out["fused"], gold)["Recall@20"]})
        in_top = [c for c in gold if c in out["graded"]]
        m["gold_kept"] = (sum(g["relevance"].get(c) in ("relevant", "partial") for c in in_top) / len(in_top)) if in_top else None
    if t["type"] == "conflict":
        m["conflict_found"] = float(conflict_hit(t, g["conflicts"]))
    if t["type"] == "missing_info" and t["missing_slots"]:
        asked = " ".join(x["item"] for x in g["missing_info"])
        m["slot_cov"] = mean(covered(s, asked) for s in t["missing_slots"])
    rel = [c for c, r in g["relevance"].items() if r == "relevant"]
    m["n_relevant"] = len(rel)
    m["relevant_precision"] = (sum(c in gold for c in rel) / len(rel)) if rel and gold else None
    return m


def run_all(system: CorrectiveRetriever, tickets: list[dict], workers: int) -> dict[str, dict]:
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip([t["id"] for t in tickets], pool.map(lambda t: system.run(t["ticket"]), tickets)))


def baseline_decisions(split: str) -> dict[str, str]:
    path = ROOT / "reports" / "week6" / f"answers_{split}.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r["answer"]["decision"] for r in rows if r["system"] == "hybrid-rag"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grader-model", default=None)
    parser.add_argument("--no-ablation", action="store_true", help="skip crag-single")
    parser.add_argument("--report-only", action="store_true", help="cached calls only")
    args = parser.parse_args()

    load_env()
    tickets = [t for t in load_tickets() if args.split == "all" or t["split"] == args.split]
    if args.limit:
        tickets = tickets[:args.limit]
    docs = load_corpus()
    store = SectionStore(docs)
    chunks = chunk_corpus(docs, 120)
    embedder = openai_embedder()
    hybrid = HybridRetriever([BM25Retriever(chunks, active_on=SNAPSHOT), DenseRetriever(chunks, embedder, SNAPSHOT)])
    llm = openai_llm()
    grader_model = args.grader_model or os.getenv("GRADER_MODEL") or llm.model
    grader_llm = llm if grader_model == llm.model else openai_llm(grader_model)
    if args.report_only:
        def no_api(*_):
            raise RuntimeError("uncached call in --report-only mode")
        llm.raw_call = grader_llm.raw_call = no_api
        embedder.raw_embed = no_api

    registry = load_conflicts()["conflicts"]
    probes = [{"id": c["id"], "ticket": c["probe_query"], "gold_sections": c["sections"]} for c in registry]

    # stage 1: analyses in parallel; stage 2: embed every query once (the embedding cache is not thread-safe)
    with ThreadPoolExecutor(args.workers) as pool:
        analyses = list(pool.map(lambda t: analyze(llm, t["ticket"]), tickets + probes))
    embedder.embed([q for a in analyses for q in a.queries])

    systems = {"crag-multi": CorrectiveRetriever(hybrid, store, llm, grader_llm=grader_llm)}
    if not args.no_ablation:
        systems["crag-single"] = CorrectiveRetriever(hybrid, store, llm, grader_llm=grader_llm, multi_query=False)
    outs = {name: run_all(s, tickets, args.workers) for name, s in systems.items()}
    probe_out = run_all(systems["crag-multi"], probes, args.workers)
    base = baseline_decisions(args.split)

    metrics = {name: {t["id"]: per_ticket(t, outs[name][t["id"]]) for t in tickets} for name in systems}
    counts = " ".join(f"{SHORT[k]}={sum(t['type'] == k for t in tickets)}" for k in TYPES)
    lines = [f"# Week-7 Corrective RAG I ({args.split} split, {len(tickets)} tickets: {counts}; analysis/rerank {llm.model}, "
             f"grader {grader_model})\n", "## Decisions (grader)\n",
             "| System | Decision acc | Macro-F1 | N acc | M acc | E acc | K acc | Conflict pair found (K) | False conflict (non-K) | Trigger acc |",
             "|---|" + "---:|" * 9]
    by_type = {k: [t for t in tickets if t["type"] == k] for k in TYPES}
    if base and all(t["id"] in base for t in tickets):
        acc = {k: avg(float(base[t["id"]] == t["expected_decision"]) for t in v) for k, v in by_type.items()}
        f1 = macro_f1([t["expected_decision"] for t in tickets], [base[t["id"]] for t in tickets])
        lines.append(f"| Week-6 Hybrid RAG (reference) | {fmt(avg(float(base[t['id']] == t['expected_decision']) for t in tickets))} | "
                     f"{fmt(f1)} | " + " | ".join(fmt(acc[k]) for k in TYPES) + " | 0.00 | - | - |")
    for name in systems:
        ms = metrics[name]
        acc = {k: avg(ms[t["id"]]["decision_ok"] for t in v) for k, v in by_type.items()}
        f1 = macro_f1([t["expected_decision"] for t in tickets], [outs[name][t["id"]]["decision"] for t in tickets])
        lines.append(f"| {name} | {fmt(avg(m['decision_ok'] for m in ms.values()))} | {fmt(f1)} | "
                     + " | ".join(fmt(acc[k]) for k in TYPES)
                     + f" | {fmt(avg(ms[t['id']].get('conflict_found') for t in by_type['conflict']))} | "
                     f"{fmt(avg(ms[t['id']]['false_conflict'] for t in tickets if t['type'] != 'conflict'))} | "
                     f"{fmt(avg(m['trigger_ok'] for m in ms.values()))} |")

    lines.append("\n## Retrieval and evidence grading\n")
    lines.append("Recall on the reranked list (normal + conflict); Both@10 on conflict tickets; gold kept = gold sections in "
                 "the graded top 10 that the grader marks relevant or partial; relevant precision = share of sections marked "
                 "relevant that are gold (a lower bound: non-gold sections can be relevant too).\n")
    lines.append("| System | Fused Recall@20 | Recall@5 | Recall@10 | K: Both@10 | M: Hit@10 | Gold kept | Relevant precision | Sections marked relevant | M slot cov |")
    lines.append("|---|" + "---:|" * 9)
    for name in systems:
        ms = metrics[name]
        nk = [ms[t["id"]] for t in tickets if t["type"] in ("normal", "conflict")]
        lines.append(f"| {name} | {fmt(avg(m.get('Recall@20_fused') for m in nk))} | {fmt(avg(m.get('Recall@5') for m in nk))} | "
                     f"{fmt(avg(m.get('Recall@10') for m in nk))} | {fmt(avg(ms[t['id']].get('Both@10') for t in by_type['conflict']))} | "
                     f"{fmt(avg(ms[t['id']].get('Hit@10') for t in by_type['missing_info']))} | "
                     f"{fmt(avg(m.get('gold_kept') for m in ms.values()))} | {fmt(avg(m.get('relevant_precision') for m in ms.values()))} | "
                     f"{fmt(avg(m['n_relevant'] for m in ms.values()))} | "
                     f"{fmt(avg(ms[t['id']].get('slot_cov') for t in by_type['missing_info']))} |")

    near = [t for t in tickets if "near_miss" in t["tags"]]
    if near:
        lines.append(f"\nNear-miss tickets (answerable, look like conflicts, n={len(near)}): " + "; ".join(
            f"{name} false conflict {sum(metrics[name][t['id']]['false_conflict'] for t in near):.0f}/{len(near)}, "
            f"decision correct {sum(metrics[name][t['id']]['decision_ok'] for t in near):.0f}/{len(near)}" for name in systems))
    rejected = Counter(r["rejected_because"] for o in outs["crag-multi"].values() for r in o["grade"]["rejected_conflicts"])
    lines.append(f"\nConflicts flagged by the LLM and refuted by the code (crag-multi): "
                 + (", ".join(f"{k} {v}" for k, v in rejected.most_common()) or "none"))
    checks = [p for o in outs["crag-multi"].values() for p in o["grade"].get("pair_checks", [])]
    if checks:
        source = Counter(c.get("source", "grader") for o in outs["crag-multi"].values() for c in o["grade"]["conflicts"])
        lines.append(f"Pairwise conflict check (crag-multi): {len(checks)} pairs examined in "
                     f"{sum(bool(o['grade'].get('pair_checks')) for o in outs['crag-multi'].values())} tickets; same attribute "
                     f"{sum(bool(c['same_attribute']) for c in checks)}, of which both sections apply "
                     f"{sum(bool(c['same_attribute'] and c['applies_a'] and c['applies_b']) for c in checks)}, of which outcomes differ "
                     f"{sum(c['conflict'] for c in checks)}. Valid conflicts by source: "
                     + ", ".join(f"{k} {v}" for k, v in source.items()))

    lines.append("\n## Planted-conflict probes (registry probe queries, crag-multi)\n")
    lines.append("| Conflict | Probe query | Both sides in top 10 | Conflict flagged with both sides | Decision |\n|---|---|---|---|---|")
    for p in probes:
        o = probe_out[p["id"]]
        both = p["gold_sections"][0] in o["graded"] and bool(set(p["gold_sections"][1:]) & set(o["graded"]))
        lines.append(f"| {p['id']} | {p['ticket']} | {'yes' if both else 'no'} | "
                     f"{'yes' if conflict_hit(p, o['grade']['conflicts']) else 'no'} | {o['decision']} |")

    lines.append("\n## Decision confusion (rows = expected, columns = predicted)\n")
    for name in systems:
        cm = confusion([t["expected_decision"] for t in tickets], [outs[name][t["id"]]["decision"] for t in tickets])
        lines.append(f"**{name}**\n\n| | " + " | ".join(DECISIONS) + " |\n|---|" + "---:|" * len(DECISIONS))
        lines += [f"| {g} | " + " | ".join(str(cm[(g, p)]) for p in DECISIONS) + " |" for g in DECISIONS]
        lines.append("")

    lines.append("## Wrong decisions (crag-multi)\n")
    lines.append("| Id | Type | Expected | Predicted | Trigger | Gold in top 10 | Ticket | Grader reason |\n|---|---|---|---|---|---|---|---|")
    for t in tickets:
        o = outs["crag-multi"][t["id"]]
        if o["decision"] != t["expected_decision"]:
            gold_in = "-" if not t["gold_sections"] else f"{sum(c in o['graded'] for c in t['gold_sections'])}/{len(t['gold_sections'])}"
            lines.append(f"| {t['id']} | {t['type']} | {t['expected_decision']} | {o['decision']} | {o['grade']['trigger']} | {gold_in} | "
                         f"{t['ticket'][:70].replace('|', '/')} | {o['grade']['reason'][:140].replace('|', '/')} |")

    u_a, u_g = llm.usage, grader_llm.usage
    calls_note = (f"{llm.model}: {u_a.calls} calls (analysis + rerank), {u_a.prompt_tokens:,} in + {u_a.completion_tokens:,} out tokens, "
                  f"USD {u_a.cost(llm.model):.3f}")
    if grader_llm is not llm:
        calls_note += (f"; grader {grader_model}: {u_g.calls} calls, {u_g.prompt_tokens:,} in + {u_g.completion_tokens:,} out, "
                       f"USD {u_g.cost(grader_model):.3f}")
    lines.append(f"\n## Cost\n\nAll calls of this run (cached ones counted at their original cost): {calls_note}.")

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = args.split + ("" if grader_model == llm.model else f"_{grader_model}")
    (OUT / f"grader_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / f"per_ticket_{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for name in systems:
            for t in tickets:
                o = outs[name][t["id"]]
                f.write(json.dumps({"system": name, "id": t["id"], "type": t["type"], "expected": t["expected_decision"],
                                    **{k: o[k] for k in ("decision", "analysis", "queries", "graded", "grade")},
                                    "metrics": metrics[name][t["id"]]}, ensure_ascii=False) + "\n")
        for p in probes:
            o = probe_out[p["id"]]
            f.write(json.dumps({"system": "probe", "id": p["id"], **{k: o[k] for k in ("decision", "analysis", "graded", "grade")}},
                               ensure_ascii=False) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
