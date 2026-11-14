# Week 6 - Reranker, shared answer prompt, Vector RAG and Hybrid RAG baselines

Data: KB from Week 3 (63 documents, 120-word chunks, validity filter on), tickets and dev/test split from Week 4, hybrid retriever from Week 5. LLM: `gpt-4o-mini`, temperature 0. Detailed tables: `reports/week6/rerank_<split>_bge-m3.md`, `reports/week6/baselines_dev.md`; answers in `reports/week6/answers_dev.jsonl`.

## 1. Weekly goals

- Add a reranker over the hybrid top 20 and check whether it brings both sides of conflicts C01 and C06 into the top 5.
- Write one answer prompt shared by every system, following `QT-MAU-PHAN-HOI-2026` and `QT-SU-DUNG-TAI-LIEU-2026`.
- Build Vector RAG and Hybrid RAG end to end and get first answer-level results on dev.

## 2. Completed work

- **LLM reranker** (`rerank.py`): one call per ticket grades the 20 hybrid candidate sections 0-3; sections are re-sorted by grade, ties keep the hybrid order, nothing is dropped. Candidates are numbered, so citations cannot be invented. The model is told to grade each section on its own and not to choose between sections that give different values, so conflicts stay visible for the Week-7 grader.
- **Cross-encoder reranker** (`cross_encoder.py`, local, no API cost): same interface and candidates; default `BAAI/bge-reranker-v2-m3` (multilingual, 568M parameters), presets for the Vietnamese models `itdainb/PhoRanker` (pyvi word segmentation) and `namdp-ptit/ViRanker`. A variant fusing the hybrid and cross-encoder rankings by RRF is also reported.
- **Shared answer prompt** (`answer.py`): decision `answer` / `clarify` / `escalate` by the rules of `QT-SU-DUNG-TAI-LIEU-2026`; reply format of `QT-MAU-PHAN-HOI-2026` (greeting, answer with conditions, next step, closing; "em" / "anh/chị"; ≤ 120 words; no document codes, no promises outside the documents, never ask for OTP/PIN/CVV). Citations go in a separate field and are checked against the context. Corrective RAG will reuse this prompt unchanged, so system differences come from the evidence, not the instructions.
- **Baselines** (`pipelines.py`): Vector RAG = dense top 5 sections → answer; Hybrid RAG = hybrid top 20 → reranker → top 5 → answer. Ablations: no reranker, cross-encoder instead of the LLM reranker.
- **Answer metrics** (`answer_metrics.py`): decision accuracy and macro-F1; gold evidence in the context (separates retrieval from generation errors); key-fact and missing-slot coverage (lexical, accent-folded; a screening signal until the Week-9 judge); gold and invalid citations; document codes shown to the customer; length.
- **Infrastructure**: cached LLM client (`llm.py`, Structured Outputs; reports rebuild with `--report-only`), scripts `eval_rerank.py` and `run_baselines.py`; 9 new tests with a fake LLM and a fake cross-encoder (48 in total).

## 3. Results / metrics

**Rerankers (retrieval, section level).** Choice made on dev; test only reported. N = normal, K = conflict, M = missing_info.

| Split | Method | N: Hit@1 | N: Recall@5 | N: MRR | K: Both@5 | M: Hit@5 | Unaccented: Recall@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| dev (45) | hybrid | 0.68 | 0.91 | 0.82 | 0.75 | 0.67 | 0.11 (n=3) |
| dev (45) | **hybrid + LLM reranker** | 0.79 | 0.96 | 0.89 | **0.88** | **0.78** | **0.83** |
| dev (45) | hybrid + cross-encoder | **0.89** | **1.00** | **0.95** | 0.62 | 0.56 | 0.44 |
| test (100) | hybrid | 0.71 | 0.85 | 0.81 | 0.47 | 0.71 | 0.81 (n=20) |
| test (100) | **hybrid + LLM reranker** | **0.82** | **0.95** | **0.89** | **0.82** | **0.82** | **0.91** |
| test (100) | hybrid + cross-encoder | 0.76 | 0.92 | 0.85 | 0.71 | 0.65 | 0.79 |
| test (100) | hybrid + cross-encoder, RRF-fused | 0.76 | 0.93 | 0.86 | 0.65 | 0.71 | 0.85 |

- **LLM reranker vs hybrid** (normal + conflict): MRR +0.07 [+0.01, +0.15] on dev, +0.06 [+0.02, +0.11] on test; test Recall@5 0.82 → 0.95. **C01 Both@5 0.00 → 1.00, C06 0.25 → 0.75** (test): the Week-5 crowding failure is fixed. Gold sections get a mean grade of 2.6, other candidates 0.3.
- **Cross-encoder**: best on normal tickets in dev, but loses conflict sides (Both@5 0.62 vs 0.88) and requirement sections for missing-info tickets; the dev MRR difference with the LLM reranker is not significant (+0.01 [−0.06, +0.08]). Test confirms it is below the LLM reranker (Recall@5 −0.05 [−0.09, −0.01], nDCG@5 −0.06 [−0.10, −0.02]; C01 Both@5 0.60) and weaker on unaccented tickets than plain hybrid. Fusion does not help.
- **Decision:** Hybrid RAG uses the LLM reranker, because keeping both sides of a conflict is what the Week-7 grader needs.
- Cost per ticket: LLM reranker ~3,000 tokens and 4 s (USD 0.08 for 145 tickets); cross-encoder no API cost but 10-13 s on CPU.

**Baselines (dev, 52 tickets: 28 normal, 9 missing_info, 7 no_evidence, 8 conflict).** Context = gold evidence among the 5 sections given to the model.

| System | Decision acc | Macro-F1 | N acc | M acc | E acc | K acc | N context | K context | N fact cov | N gold cited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vector RAG | 0.56 | 0.28 | 1.00 | 0.00 | 0.14 | 0.00 | 0.89 | 0.75 | 0.85 | 0.93 |
| **Hybrid RAG** (LLM reranker) | **0.58** | **0.35** | 1.00 | 0.11 | 0.14 | 0.00 | 0.93 | **0.88** | **0.92** | **0.96** |
| Hybrid RAG, no reranker | 0.58 | 0.35 | 1.00 | 0.11 | 0.14 | 0.00 | 0.86 | 0.75 | 0.90 | 0.86 |
| Hybrid RAG, cross-encoder | 0.58 | 0.35 | 1.00 | 0.11 | 0.14 | 0.00 | **1.00** | 0.62 | 0.90 | **0.96** |

- **The baselines almost always answer** (Hybrid RAG: 50 of 52 tickets). Normal tickets are trivially correct, but only 1/9 missing-info tickets get a question, 1/7 no-evidence tickets are escalated and **0/8 conflicts are detected**, although both sides were in the context for 7/8.
- Typical errors: K017 gives "3-5 working days" from one document and ignores the other; M017 explains the 30-day rule instead of asking for the delivery date; 3 of 7 no-evidence replies contain unsupported claims (E014 "about 34 kg", not in the catalogue; E020 invents a paid screen-protector service); 3 others say the information is missing but still set decision "answer".
- Better retrieval improves the evidence (context, gold citations) but does not change the decisions: retrieval alone does not make the model ask, refuse or escalate.
- Format: mean 58 words, none over 120, no document code shown to customers; at most 1 invalid citation per system (M008, mistyped section id). Answers: ~1,500 tokens and 2.4 s per ticket. All Week-6 API runs cost about USD 0.17.

## 4. Issues & resolutions

| Issue | Resolution |
|---|---|
| The sandbox cannot reach the OpenAI API or download models | Code tested offline with fakes; real runs on the local machine; LLM calls and cross-encoder scores cached |
| The LLM reranker grades some conflict sides 0: narrow-scope notices (TB-BAO-HANH, TB-GIAO-HANG) and the credit-card instalment section (K005, K008, K010, K011, K015, K016) | Accepted, so the reranker never resolves conflicts itself; the Week-7 grader gets the top 10 and multi-query retrieval |
| Cross-encoder drops conflict sides and is slow on CPU | Kept as a documented alternative; PhoRanker / ViRanker can be tried with `--ce-model` |
| Baselines answer when they should ask or escalate | Expected for plain RAG and the gap Corrective RAG must close: the decision moves to an explicit evidence-grading step (Week 7) |
| Replies that refuse in the text but set decision "answer"; lexical key-fact coverage | Decision accuracy kept strict; the reply text is scored by the LLM judge in Week 9 |
| Dev has 52 tickets (Week-4 text said 51) | Counts taken from `tickets.jsonl`; 45 have gold sections and are used for retrieval metrics |

## 5. Plan for next week (Week 7)

- Ticket analysis: sub-questions, product/model, dates and order data given or missing.
- Multi-query retrieval: one query per sub-question plus the ticket, fused by RRF, reranked by the LLM reranker.
- Evidence grader over the top 10: relevance, sufficiency, missing customer information, conflicts between valid documents; unit tests on the planted conflicts and near-misses.
- Measure the grader's decisions on dev against `expected_decision`, with the Week-6 baselines as reference (decision acc 0.58, K acc 0.00, E acc 0.14).

Reproduce: `pip install -r requirements-ce.txt`, then `python scripts/eval_rerank.py --split dev|test --rerankers llm,ce` and `python scripts/run_baselines.py --ce bge-m3` (add `--report-only` to rebuild without API calls).
