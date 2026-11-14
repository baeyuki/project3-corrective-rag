# 12-week plan

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Survey of similar applications and related research; scope; data formats; feasibility checks | `week1.md`, data formats, tokenisation and grading proofs of concept |
| 2 | Knowledge base I: policies, procedures, FAQ, troubleshooting guides | ~25 documents in the agreed format |
| 3 | Knowledge base II: product catalogs, one expired policy, planted conflicts; parsing and chunking | ~50 documents, ~1,000-1,500 chunks, corpus statistics |
| 4 | Evaluation set: ~150 labelled tickets (normal, missing info, no evidence, conflict); label checks; dev split | `tickets.jsonl`, label validation |
| 5 | Retrieval: BM25, embeddings, hybrid fusion (RRF); retrieval metrics | Recall@K, MRR, nDCG per method |
| 6 | Reranker; shared answer prompt; Vector RAG and Hybrid RAG baselines | Two runnable baselines |
| 7 | Corrective RAG I: ticket analysis, multi-query retrieval, evidence grading | Grading step with tests |
| 8 | Corrective RAG II: query rewriting, decision branches, citation checks | Complete Corrective RAG |
| 9 | Metrics, LLM-as-judge, evaluation runner and report | Automatic comparison report |
| 10 | Error analysis and tuning on the dev split; full evaluation; results on tickets unused for tuning | Final results and error analysis |
| 11 | FastAPI service, Streamlit UI, Docker; latency and cost measurements | Dockerised demo |
| 12 | Final report, slides, demo | Report, slides, final code |

## Revisions

- **Week 3 (2026-10-22):** document target exceeded (63, of which 13 simulated operating documents). The chunk estimate of 1,000-1,500 is replaced by measured values: 275 sections, 304 chunks at the default 120-word limit, 364 at 60 words. Product data are taken only from official manufacturer pages, so the corpus is not padded to reach a chunk count; see [week3.md](week3.md), section 4.
- **Week 4 (2026-10-22):** 168 tickets instead of ~150, so that missing_info, no_evidence and conflict each have at least 20 tickets while normal tickets stay the majority (55%). Dev split: 51 tickets (30% per type, grouped by evidence and by conflict).
- **Week 6 (2026-11-14):** reranker implemented with an LLM (gpt-4o-mini, grades 0-3 over the hybrid top 20); a local cross-encoder (BAAI/bge-reranker-v2-m3, optional PhoRanker / ViRanker) is added as an alternative and run on the local machine, since the sandbox cannot download models. Hybrid RAG = hybrid RRF → reranker → top 5; Vector RAG = dense top 5. Answer-level results on dev only; test kept for Week 10.
