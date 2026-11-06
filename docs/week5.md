# Week 5 - Retrieval: BM25, embeddings, hybrid fusion (RRF) and retrieval metrics

**Goal:** implement the three first-stage retrievers that the baselines and Corrective RAG will share, and measure them on the Week-4 tickets with Recall@K, MRR and nDCG.

**Status:** complete. Five retrievers (three BM25 tokenisers, vector, hybrid) × three chunk granularities evaluated on dev (45 tickets) and test (100 tickets). Embeddings were computed once with `text-embedding-3-small` on 2026-10-22 (605 cached texts: all chunks at three granularities plus the tickets).

**Decision for Weeks 6-8:** first-stage retrieval = **hybrid RRF of accent-folded BM25 and `text-embedding-3-small`, 120-word chunks, validity filter on**, passing the top 10-20 sections to the next stage.

## 1. Deliverables

| Deliverable | Location |
|---|---|
| Retrievers: BM25, dense, hybrid RRF; validity filter; chunk → section ranking | `src/rag_cskh/retrieval.py` |
| Embeddings with on-disk cache (OpenAI `text-embedding-3-small` by default) | `src/rag_cskh/embeddings.py` |
| Metrics: Hit@K, Recall@K, All@K, MRR, nDCG@K, Both@K; bootstrap CIs | `src/rag_cskh/metrics.py` |
| Evaluation runner (tables, per-ticket CSV for error analysis) | `scripts/eval_retrieval.py` → `reports/week5/` |
| Tests (38 in total, 6 new) | `tests/test_retrieval.py` |

## 2. Design

**Retrievers.**

- **BM25** (Robertson & Zaragoza 2009), k1 = 1.5, b = 0.75, over the chunk text including its "document > section" header. Three tokenisers are compared: whitespace words; syllables + syllable bigrams; syllables + bigrams + accent-folded copies (the Week-1 tokeniser, used as "bm25").
- **Vector:** cosine similarity between L2-normalised embeddings of the ticket and of each chunk. Model from `EMBED_MODEL` (default `text-embedding-3-small`, multilingual). Every text is embedded once and cached in `data/index/embeddings/`, so re-runs and granularity changes cost nothing.
- **Hybrid:** Reciprocal Rank Fusion (Cormack et al. 2009) of the top 50 BM25 and top 50 vector chunks, `score = Σ 1/(60 + rank)`. RRF needs no score calibration between a BM25 score and a cosine, which is why it was chosen over weighted score sums.

**Evaluation choices.**

- **Citation level.** Chunk rankings are collapsed to section rankings (first chunk of a section decides its rank), so the three granularities and all methods are compared on the same unit as the gold labels.
- **Validity filter.** Chunks of documents not usable on 2026-10-22 are removed after scoring, identically for all methods. The filter is metadata, not ranking, and is reported separately (section 4).
- **Queries are the raw tickets.** Query rewriting and multi-query retrieval belong to Corrective RAG (Weeks 7-8).
- **Tickets evaluated:** normal and conflict (answer evidence), missing_info (the section stating the required information). no_evidence tickets have no gold sections and are evaluated in Week 7 through the grader's decision.
- **Both@K** for conflict tickets: both sides of the conflict must be in the top K; a grader cannot detect a conflict whose second side it never receives. **All@K** is the analogue for multi-document answers.
- **Protocol.** Choices are made on the dev split (45 evaluated tickets). Test numbers (100 tickets) are recorded as baselines; nothing in this week was tuned on them. 95% intervals come from 2,000 bootstrap resamples of tickets.

## 3. Results

Numbers below use the final Week-4 labels (after the LLM cross-check, see `week4.md`).

**Dev split** (normal + conflict, 120-word chunks, n = 36): hybrid has the highest MRR and Recall@20 (MRR 0.82 [0.72, 0.90], Recall@20 1.00) against BM25 (0.77 [0.67, 0.88], 0.97). The paired difference hybrid − BM25 is +0.04 MRR [−0.01, +0.10]: same direction, not yet significant on 36 tickets. The choice of hybrid was made on this evidence and on the design argument (it combines two complementary signals); the test split below confirms it.

**Test split, main table** (N = normal, K = conflict, M = missing_info; 120-word chunks; other granularities in `reports/week5/summary_test.md`):

| Method | N: Hit@1 | N: Recall@5 | N: All@5 | N: MRR | N: nDCG@10 | K: Both@5 | K: Both@10 | M: Hit@5 | M: MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25-words | 0.48 | 0.78 | 0.71 | 0.65 | 0.67 | 0.41 | 0.47 | 0.53 | 0.41 |
| bm25-syllable | 0.50 | 0.74 | 0.68 | 0.65 | 0.67 | 0.47 | 0.53 | 0.53 | 0.30 |
| bm25 (folded) | 0.56 | 0.82 | 0.74 | 0.72 | 0.74 | **0.59** | 0.59 | 0.59 | 0.36 |
| vector | 0.64 | 0.80 | 0.74 | 0.72 | 0.73 | 0.53 | 0.76 | 0.59 | 0.46 |
| **hybrid** | **0.71** | **0.85** | **0.79** | **0.81** | **0.81** | 0.47 | **0.88** | **0.71** | **0.48** |

**Test split, Recall@K curve** (normal + conflict, n = 83):

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| bm25-words | 0.35 | 0.65 | 0.74 | 0.79 | 0.85 | 0.67 [0.58, 0.74] |
| bm25-syllable | 0.37 | 0.65 | 0.72 | 0.81 | 0.87 | 0.67 [0.59, 0.75] |
| bm25 | 0.44 | 0.74 | 0.81 | 0.88 | 0.94 | 0.76 [0.69, 0.83] |
| vector | 0.48 | 0.68 | 0.78 | 0.84 | 0.92 | 0.74 [0.67, 0.82] |
| hybrid | **0.53** | **0.79** | **0.82** | **0.94** | **0.96** | **0.84 [0.77, 0.90]** |

**Paired differences against BM25** (same 83 test tickets, 95% bootstrap CI):

| Method | Δ MRR | Δ Recall@5 |
|---|---|---|
| vector | −0.02 [−0.10, +0.06] | −0.03 [−0.10, +0.05] |
| hybrid | **+0.07 [+0.01, +0.13]** | +0.01 [−0.05, +0.07] |

**Test split, Recall@5 by tag:**

| Tag | n | bm25-words | bm25-syllable | bm25 | vector | hybrid |
|---|---:|---:|---:|---:|---:|---:|
| no_diacritics | 20 | 0.43 | 0.48 | **0.84** | 0.70 | 0.81 |
| multi_doc | 31 | 0.61 | 0.61 | **0.73** | 0.64 | 0.70 |
| catalog | 21 | 0.92 | 0.90 | 0.90 | 0.94 | **0.97** |
| manufacturer | 8 | 0.67 | 0.73 | 0.85 | 0.85 | 0.85 |
| near_miss | 9 | **0.67** | 0.56 | 0.56 | 0.39 | 0.56 |
| expired_trap | 4 | 0.75 | **0.88** | **0.88** | 0.75 | 0.75 |

**Granularity:** for every method the three granularities are within 0.04 MRR on test (hybrid normal-ticket MRR: 0.79 section, 0.81 at 120 words, 0.78 at 60 words). 120 words stays the default.

## 4. Findings

1. **Hybrid is the best first stage, and the only significant improvement over BM25.** It raises MRR from 0.76 to 0.84 (+0.07 [+0.01, +0.13]), Hit@1 on normal tickets from 0.56 to 0.71, Recall@10 from 0.88 to 0.94, and the retrieval of the requirement sections for missing_info tickets (MRR 0.36 → 0.48). Recall@5 barely moves: the gain is in **ordering** the right section first, which matters because the answer prompt will read only the top few sections.
2. **Vector alone is not better than BM25** (−0.02 MRR, interval spans zero). The two fail on different tickets, which is why fusion works. Vector fixes paraphrase failures: "Công ty em cần xuất hóa đơn VAT…" (BM25 rank 5 → vector rank 1), "don cua em toi dau roi shop" (rank 13 → 1), the fridge-fit question "Hốc tủ … có đặt vừa không" (rank 7 → 1), the Daikin warranty start date (rank 4 → 1). BM25 wins on exact terms and unaccented text: "laptop 25 trieu em muon tra tien mat khi nhan hang" (BM25 rank 2, vector not in top 20), "Chủ nhật … gửi bảo hành" (rank 1 vs 12).
3. **Accent folding remains essential.** On unaccented tickets, embeddings reach 0.70 Recall@5 against 0.84 for folded BM25; the hybrid keeps 0.81. Without folding, BM25 drops to 0.43.
4. **Conflicts are the open problem, and they fail differently by depth.** Both sides of a conflict are in the top 5 for 59% of test conflict tickets with BM25 and only 47% with hybrid, but in the top 10 for 88% with hybrid against 59% with BM25. Per conflict (Both@5, test): C01 promotion 0/5 for every method, C02 3/4, C03 4/4, C06 3/4 with BM25 but 1/4 with hybrid. Two causes:
   - *Same-document crowding (C01):* the four sections of the promotion terms fill the top 5 and push out the policy section for every method.
   - *Semantic drift (C06):* the embedding prefers the installation price list and the moving FAQ over the policy sentence "vật tư phát sinh … báo giá riêng", so fusion demotes the policy side.
   Consequence for Weeks 6-7: the grader must see the top 10, not the top 5, or the candidates must be reranked or diversified before truncation.
5. **Near-misses are hurt by embeddings** (Recall@5 0.39 for vector). These tickets mention the vocabulary of a conflict (promotion, 15 days, free delivery) while the evidence is a scope sentence; the embedding retrieves the conflicting topic instead. This is exactly the case in which a conflict-detecting grader must not over-trigger.
6. **missing_info requirement sections remain hard** (best Hit@5 0.71). Short requests such as "Hủy đơn giúp em" match many sections weakly; the Week-7 grader should be allowed to ask for information when the order-handling procedure is among the candidates, not only at rank 1.

A per-document cap (at most two sections per document in the top K) was tried with BM25 on dev to counter crowding: Both@5 rose from 0.75 to 0.88 (one ticket) but Recall@5 fell from 0.90 to 0.88. It was not adopted; reranking (Week 6) and multi-query retrieval (Week 7) address crowding without discarding relevant sections.

## 5. Validity filter

Without the filter, the expired policy `CS-DOI-TRA-2024` appears in the BM25 top 5 for 5-7% of the evaluated tickets. Because it shares section ids with the active policy, it rarely displaces the gold section (test MRR on normal tickets 0.71 without the filter vs 0.72 with it), but it would be passed to the LLM as evidence with contradictory values. The filter is applied by default for all methods; the unfiltered configuration is kept as an ablation for Week 9 (`--no-filter`).

## 6. Implications for Weeks 6-8

- **Candidate depth:** hybrid Recall@20 is 0.96 on test, so a reranker over the top 20 sections has the evidence in its candidate set for almost every ticket; the problem is ordering.
- **Conflicts:** pass the top 10 to the grader (Both@10 = 0.88 with hybrid), and test in Week 6 whether a cross-encoder reranker can bring the policy side of C01 and C06 into the top 5.
- **Reporting:** keep the per-tag tables; averages hide the accented/unaccented gap, the multi-document gap and the near-miss weakness of embeddings.

## 7. Reproducing

```bash
python scripts/check_env.py --ping
python scripts/eval_retrieval.py --dense --split dev
python scripts/eval_retrieval.py --dense --split test
python scripts/eval_retrieval.py --split test --no-filter      # validity-filter ablation (BM25)
```

Embeddings are cached in `data/index/embeddings/text-embedding-3-small.jsonl` (git-ignored, 605 texts). Re-running costs nothing; a fresh run costs well under USD 0.01. Per-ticket rankings for error analysis are in `reports/week5/per_ticket_<split>.csv`.
