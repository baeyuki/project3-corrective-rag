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

