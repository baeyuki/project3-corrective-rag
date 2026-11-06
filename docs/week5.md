# Week 5 - Retrieval: BM25, embeddings, hybrid fusion (RRF) and retrieval metrics

Data: KB from Week 3 (63 documents, 120-word chunks), tickets and dev/test split from Week 4. Detailed tables: `reports/week5/summary_<split>.md`.

## 1. Weekly goals

- Implement the first-stage retrievers that the baselines and Corrective RAG will share: BM25, embedding (vector) search and hybrid fusion.
- Measure them on the labelled tickets at citation level (`doc_id#section-id`) with Recall@K, MRR and nDCG, plus a metric for conflict tickets.
- Choose the configuration on dev and record test results as baselines.

## 2. Completed work

- **BM25** (k1 = 1.5, b = 0.75) with three tokenisers: whitespace words; syllables + bigrams; syllables + bigrams + accent-folded copies (the Week-1 tokeniser).
- **Vector retrieval** with OpenAI `text-embedding-3-small` and cosine similarity. Every text is embedded once and cached on disk (605 texts: chunks at three granularities and the tickets), so re-runs cost nothing.
- **Hybrid retrieval**: Reciprocal Rank Fusion of the top 50 BM25 and top 50 vector chunks (`Σ 1/(60 + rank)`); RRF needs no calibration between BM25 scores and cosines.
- **Validity filter**: chunks of documents not usable on 2026-10-22 (the expired `CS-DOI-TRA-2024`) are removed for every method.
- **Metrics** (`metrics.py`): Hit@K, Recall@K, All@K (all gold sections found), MRR, nDCG@K and **Both@K** (both sides of a conflict in the top K, since a grader cannot detect a conflict it never sees); 95% bootstrap intervals and paired differences.
- **Evaluation runner** (`eval_retrieval.py`): chunk rankings are collapsed to section rankings so all methods and granularities are compared on the gold unit; per-ticket CSVs for error analysis.
- Queries are the raw tickets (query rewriting belongs to Corrective RAG). no_evidence tickets have no gold sections and are evaluated in Week 7 through the grader's decision.
- 6 new tests (39 in total).

## 3. Results / metrics

Evaluated tickets: 45 dev, 100 test (normal, conflict and missing_info; 120-word chunks).

**Test split** (N = normal, K = conflict, M = missing_info):

| Method | N: Hit@1 | N: Recall@5 | N: MRR | K: Both@5 | K: Both@10 | M: Hit@5 | MRR N+K [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25-words | 0.48 | 0.78 | 0.65 | 0.41 | 0.47 | 0.53 | 0.67 [0.58, 0.74] |
| bm25-syllable | 0.50 | 0.74 | 0.65 | 0.47 | 0.53 | 0.53 | 0.67 [0.59, 0.75] |
| bm25 (folded) | 0.56 | 0.82 | 0.72 | **0.59** | 0.59 | 0.59 | 0.76 [0.69, 0.83] |
| vector | 0.64 | 0.80 | 0.72 | 0.53 | 0.76 | 0.59 | 0.74 [0.67, 0.82] |
| **hybrid** | **0.71** | **0.85** | **0.81** | 0.47 | **0.88** | **0.71** | **0.84 [0.77, 0.90]** |

- **Hybrid is the best first stage**: MRR +0.07 over BM25 [+0.01, +0.13], the only significant difference; Recall@10 0.94 and Recall@20 0.96. Vector alone is not better than BM25 (−0.02 [−0.10, +0.06]). The two fail on different tickets: vector fixes paraphrases ("xuất hóa đơn VAT", "don cua em toi dau roi"), BM25 wins on exact terms and unaccented text.
- **Dev confirms the choice** (MRR hybrid 0.82 vs BM25 0.77). The decision was made on dev; test was not used for tuning.
- **Accent folding is essential**: on unaccented tickets Recall@5 is 0.84 for folded BM25, 0.70 for vector and 0.43 for BM25 without folding.
- **Chunk granularity hardly matters**: section, 120 words and 60 words are within 0.04 MRR for every method. 120 words stays the default.
- **Validity filter**: without it, the expired policy appears in the BM25 top 5 for 5-7% of tickets; it rarely displaces the gold section, but it would reach the LLM as contradictory evidence.
- **Decision for Weeks 6-8:** hybrid RRF (folded BM25 + `text-embedding-3-small`), 120-word chunks, validity filter on, top 10-20 sections passed to the next stage.

## 4. Issues & resolutions

| Issue | Resolution |
|---|---|
| Conflicts are hard to retrieve completely: hybrid finds both sides in the top 5 for only 47% of test conflict tickets (88% in the top 10). In C01 the four promotion sections fill the top 5 and push out the policy; in C06 the embedding prefers the installation price list over the policy sentence | The grader in Week 7 receives the top 10, not the top 5; a reranker (Week 6) and multi-query retrieval (Week 7) are to fix the ordering |
| A per-document cap (max 2 sections per document) improved Both@5 on dev by one ticket but lowered Recall@5 | Not adopted; reranking addresses crowding without dropping relevant sections |
| Embeddings hurt near-miss tickets (Recall@5 0.39 for vector): they retrieve the conflicting topic instead of the scope sentence | Reported per tag; this is the case where the Week-7 conflict grader must not over-trigger |
| Requirement sections for missing_info tickets are hard to find for short requests ("Hủy đơn giúp em"), best Hit@5 0.71 | The Week-7 grader may ask for information when the order-handling procedure is anywhere in the candidates, not only at rank 1 |
| The sandbox could not reach the OpenAI API; the API key expired once | Embedding run on the local machine with a new key in `.env` (git-ignored); results cached so all later runs need no API calls |
| Week-4 labels changed after the LLM cross-check (2 tickets) | Evaluation re-run on the final labels from the cached embeddings; conclusions unchanged |

## 5. Plan for next week (Week 6)

- Add a reranker (cross-encoder or LLM-based) over the hybrid top 20 and test whether it brings both sides of C01 and C06 into the top 5.
- Write one shared answer prompt following `QT-MAU-PHAN-HOI-2026` (Vietnamese reply, citations `doc_id#section-id`).
- Build the two baselines, Vector RAG and Hybrid RAG, end to end (retrieve → answer) and run them on dev to get first answer-level results.

Reproduce: `python scripts/eval_retrieval.py --dense --split dev` and `--split test` (cached embeddings, no cost); `--no-filter` for the validity-filter ablation.
