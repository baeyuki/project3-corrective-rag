# Corrective RAG for Customer-Support Tickets - Project III

An assistant for customer-support agents of a fictional Vietnamese electronics retailer. For each ticket it retrieves evidence from internal documents and drafts a cited reply, asks the customer for missing information, or escalates when the documents have no grounds or contradict each other. Documents, tickets and all replies are in Vietnamese. Corrective RAG will be compared with Vector RAG and Hybrid RAG on a labelled ticket set.

## Status

- **Week 1:** survey of similar applications and related research, scope, data formats, feasibility checks: [docs/week1.md](docs/week1.md)
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
pytest -q
```

## Layout

```
docs/week1.md, plan.md        survey and plan
data/kb/_TEMPLATE.md           knowledge-base document format
data/eval/ticket_schema.json   ticket label format (+ examples.jsonl)
data/poc/                      toy corpus and queries for the feasibility checks
src/rag_cskh/                  text.py (Vietnamese tokeniser), bm25.py
scripts/                       check_env.py, poc_tokenization.py, poc_grading.py
tests/                         data formats, tokeniser, BM25
```
