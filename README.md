# Corrective RAG for Customer-Support Tickets - Project III

An assistant for customer-support agents of a fictional Vietnamese electronics retailer. For each ticket it retrieves evidence from internal documents and drafts a cited reply, asks the customer for missing information, or escalates when the documents have no grounds or contradict each other. Documents, tickets and all replies are in Vietnamese. Corrective RAG will be compared with Vector RAG and Hybrid RAG on a labelled ticket set.

## Status

- **Week 1:** survey of similar applications and related research, scope, data formats, feasibility checks: [docs/week1.md](docs/week1.md)
- **Week 2:** 25-document Vietnamese knowledge base covering policies, procedures, FAQs and troubleshooting: [docs/week2.md](docs/week2.md)
- Week-2 provenance, official web sources and controlled assumptions: [docs/week2_sources.md](docs/week2_sources.md)
- **Week 3:** 63-document knowledge base: 35 real product models from official manufacturer pages, manufacturer warranties, one expired policy, six planted conflicts, 13 simulated operating documents (fees, membership, extended warranty, installation, safety); parser, section-aware chunker and corpus statistics: [docs/week3.md](docs/week3.md)
- Week-3 sources for every product and warranty fact: [docs/week3_sources.md](docs/week3_sources.md); conflict registry: `data/eval/kb_conflicts.json`
- **Week 4:** 168 labelled tickets (94 normal, 26 missing info, 23 no evidence, 25 conflict) with tags, grouped dev/test split and automatic label checks: [docs/week4.md](docs/week4.md); label guide: [docs/week4_label_guide.md](docs/week4_label_guide.md)
- **Week 5:** BM25 / vector / hybrid-RRF retrievers with validity filter, citation-level Recall@K, MRR, nDCG, Both@K and bootstrap CIs; hybrid (BM25 + text-embedding-3-small) is best, test MRR 0.84 vs 0.76 for BM25: [docs/week5.md](docs/week5.md)
- Plan: [docs/plan.md](docs/plan.md)

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate       # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
copy .env.example .env                               # then set OPENAI_API_KEY
python scripts/check_env.py --ping                   # versions, API key, one tiny API call
python scripts/poc_tokenization.py                  # BM25 tokenisation check (offline)
python scripts/poc_grading.py                       # LLM evidence-grading check (API, < $0.01)
python scripts/build_corpus.py                      # parse + chunk data/kb, print corpus statistics
python scripts/probe_conflicts.py                   # BM25 ranks of planted conflicts (offline)
python scripts/validate_tickets.py                  # label checks + dataset report for data/eval/tickets.jsonl
python scripts/split_tickets.py --check              # verify the stored dev/test split
python scripts/llm_label_check.py                    # blind LLM cross-check of the ticket labels (JUDGE_MODEL, ~USD 0.5-1)
python scripts/eval_retrieval.py --split dev         # Week-5 BM25 evaluation (offline)
python scripts/eval_retrieval.py --dense --split dev # + vector and hybrid (OpenAI embeddings, cached, < $0.01)
pytest -q
```

## Layout

```
docs/week1.md, plan.md        survey and plan
data/kb/*.md                   knowledge-base documents (CS, QT, FAQ, HD, SP, NSX, KM, TB, BG, HT) and authoring template
data/eval/tickets.jsonl        168 labelled tickets with dev/test split (Week 4)
data/eval/ticket_schema.json   ticket label format (+ examples.jsonl, Week-1 format examples)
data/eval/kb_conflicts.json    planted conflicts, expired policy, compatible near-misses
data/index/                    built chunks and statistics (generated, git-ignored)
data/poc/                      toy corpus and queries for the feasibility checks
src/rag_cskh/                  text.py (Vietnamese tokeniser), bm25.py, corpus.py (parser), chunking.py, tickets.py (label validation),
                               retrieval.py, embeddings.py, metrics.py
scripts/                       check_env.py, poc_tokenization.py, poc_grading.py, build_corpus.py, probe_conflicts.py,
                               validate_tickets.py, split_tickets.py, llm_label_check.py, eval_retrieval.py
tests/                         data formats, knowledge base, chunking, tickets, retrieval and metrics, tokeniser, BM25
```
