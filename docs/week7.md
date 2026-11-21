# Week 7 - Corrective RAG I: ticket analysis, multi-query retrieval, evidence grading

Data: KB from Week 3, tickets and dev split from Week 4 (52 dev tickets: 28 normal, 9 missing_info, 7 no_evidence, 8 conflict), hybrid retriever (Week 5) and LLM reranker (Week 6). All tuning on dev; the test split is untouched until Week 10. Detailed tables: `reports/week7/grader_dev_gpt-4.1-mini.md` (final), earlier versions `grader_dev_v1.md` to `grader_dev_v3*.md`; per-ticket judgments in `per_ticket_dev*.jsonl`.

## 1. Weekly goals

- Analyse each ticket into self-contained sub-questions and the facts the customer gave.
- Retrieve with several queries per ticket and fuse the results.
- Build an evidence grader that decides answer / clarify / escalate and detects conflicts, with tests on the planted conflicts and near-misses.

## 2. Completed work

- **Ticket analysis** (`analysis.py`): one LLM call rewrites the ticket with full diacritics and returns 1-3 sub-questions the customer actually asks (main question first, the customer's own case kept), the facts given and the products named. It only restates the ticket.
- **Multi-query retrieval** (`multiquery.py`): the raw ticket, the normalized ticket and each sub-question go through the hybrid retriever; section rankings are fused by RRF (top 20), reranked by the Week-6 LLM reranker, and the top 10 go to the grader. The raw ticket is always a query, so multi-query can only add candidates.
- **Evidence grader** (`grader.py`), "the LLM judges, the code decides":
  - main call: relevance of each section, status of each sub-question (answered / needs customer information / not in the documents), missing customer information, conflicts;
  - pairwise conflict check: every pair of relevant/partial sections from different documents is examined separately: same attribute of the same service? does each section apply to this kind of case? what outcome does each give for this customer, and are the two outcomes incompatible? A pair is a conflict only if all hold. This follows the registry's definition (C02, order 400,000 d at 5 km: one section makes delivery free, the other does not; near-miss N01, order ≥ 500,000 d within 10 km: both make it free);
  - the code rejects impossible conflicts (same document, a side graded irrelevant, out of scope, identical or empty outcomes) and applies `QT-SU-DUNG-TAI-LIEU-2026` in priority order: conflict → escalate; main question not in the documents → escalate; main question needs customer information → clarify; otherwise answer.
- **Pipeline** (`crag.py`): analyse → multi-query → rerank → grade. Ablation `crag-single` uses the raw ticket only.
- **Evaluation** (`eval_grader.py`): decisions against `expected_decision`, conflict pairs found, false conflicts, near-miss tickets, evidence kept by the grader, missing-slot coverage, and the six registry probe queries (`kb_conflicts.json`). Reference: Week-6 Hybrid RAG decisions.
- 12 new offline tests (60 in total): decision order, planted conflict C01 → escalate, near-miss pairs N01/N02 → no conflict, refuted conflicts, pair construction, prompts without labels, end-to-end stage with a fake LLM.

## 3. Results / metrics

**Iterations on dev** (crag-multi; N/M/E/K = decision accuracy by ticket type; false conflict = share of non-conflict tickets escalated as conflict; probes = registry conflicts flagged with both sides):

| Version | Grader | Decision acc | Macro-F1 | N | M | E | K | False conflict | Probes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Week-6 Hybrid RAG | - | 0.58 | 0.35 | 1.00 | 0.11 | 0.14 | 0.00 | - | - |
| v1: one holistic call | gpt-4o-mini | 0.56 | 0.55 | 0.54 | 0.78 | 0.86 | 0.12 | 0.00 | 0/6 |
| v2: main question decides, pair check (scope) | gpt-4o-mini | 0.65 | 0.60 | 0.75 | 0.56 | 0.71 | 0.38 | 0.09 | 1/6 |
| v3: pair check (applies + outcome) | gpt-4o-mini | 0.62 | 0.58 | 0.68 | 0.78 | 0.71 | 0.12 | 0.11 | 1/6 |
| v3 | gpt-4.1-mini | 0.67 | 0.68 | 0.64 | 0.67 | 0.43 | 1.00 | 0.32 | 5/6 |
| v4: incompatible outcomes only | gpt-4o-mini | 0.71 | 0.68 | 0.75 | 0.78 | 0.71 | 0.50 | 0.07 | 4/6 |
| **v4 (final)** | **gpt-4.1-mini** | **0.79** | **0.77** | 0.82 | 0.78 | 0.43 | **1.00** | 0.11 | **5/6** |

- What each iteration fixed: v1 asked the customer whenever any (often invented) sub-question lacked information (12/28 normal tickets) and almost never filled the conflict list; v2 let only the main question decide and added the pairwise check; in v2 the model read "case in scope" as "customer qualifies" and rejected 6 of 8 true conflicts; with gpt-4.1-mini, v3 flagged complementary sections as conflicts (19 of 23 pair conflicts false); v4 requires incompatible outcomes.
- **Final configuration (v4, grader gpt-4.1-mini):** decision accuracy 0.79 vs 0.58 for Hybrid RAG, macro-F1 0.77 vs 0.35. All 8 conflict tickets are escalated with the registered pair; missing-info tickets get a question in 7/9 (Hybrid RAG 1/9). The smaller gpt-4o-mini cannot do the conflict check reliably (K 0.50).
- **Multi-query vs single query** (final grader): both sides of every conflict in the top 10 (Both@10 1.00 vs 0.88), Recall@10 1.00 vs 0.97, missing-slot coverage 0.74 vs 0.54, decision accuracy 0.79 vs 0.77.
- **Evidence grading:** every gold section in the top 10 is kept as relevant or partial; 76% of sections marked relevant are gold (a lower bound); 1.7 sections marked relevant per ticket.
- **Cost:** ~USD 0.003 per ticket (analysis and reranking with gpt-4o-mini, grading with gpt-4.1-mini); all Week-7 runs together under USD 1.
- These are dev results after four prompt iterations on the same tickets, so they are optimistic; the configuration is now frozen and will be measured on test in Week 10.

## 4. Issues & resolutions

| Issue | Resolution |
|---|---|
| No-evidence tickets answered from absence: E014 (weight not in the catalogue) and E019 (timer not listed among features) are "answered"; E015 answered from an unrelated delivery rule (E acc 0.43) | Week 8 citation check: an answer must cite a section that states the value asked for; otherwise escalate |
| False conflicts from the holistic main call where both sections give the same outcome for the customer (N012: order 12,000,000 d at 8 km, free under both rules; N022: 5,000,000 d above both instalment thresholds), and from sections on related services (M012, M020) | Week 8: conflicts from the main call must be confirmed by the pairwise check on the same pair |
| Probe C01 missed by gpt-4.1-mini: the policy section was not among the graded relevant sections, so the pair was never checked | Week 8: when the main call grades a section partial or irrelevant but it states the same attribute as a relevant one, include it in the pair check |
| gpt-4o-mini cannot run the conflict check reliably | Grader set to gpt-4.1-mini (`GRADER_MODEL`); analysis and reranking stay on gpt-4o-mini |
| `JUDGE_MODEL` (Week 9 LLM judge) is also gpt-4.1-mini, the grader model | Week 9 uses a different judge model so that the system is not judged by its own grader |
| Prompts were tuned on the dev tickets | Configuration frozen at v4; test split kept for Week 10; every iteration kept in `reports/week7/` |

## 5. Plan for next week (Week 8)

- Decision branches: answer with the Week-6 shared prompt using the graded evidence; clarify with the grader's missing items (at most three); escalate with the department and response time from the complaints policy.
- Query rewriting: when the main question is not in the documents or no section is relevant, rewrite the query once and grade again before escalating.
- Citation check: each fact in the reply must be supported by a cited section that states it; failed checks lead to a retry or an escalation.
- Fixes from section 4 (pairwise confirmation of conflicts, pair check for same-attribute sections).
- Run the complete Corrective RAG on dev and compare with Vector RAG and Hybrid RAG on decisions and answers.

Reproduce: `python scripts/eval_grader.py --grader-model gpt-4.1-mini` (add `--report-only` to rebuild from the cache without API calls).
