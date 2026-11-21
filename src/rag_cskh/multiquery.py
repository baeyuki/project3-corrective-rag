"""Multi-query retrieval (Corrective RAG step 2, Week 7).

Every query of the ticket analysis (raw ticket, normalized ticket, sub-questions) is run through the Week-5 hybrid
retriever; the section rankings are fused with Reciprocal Rank Fusion, so a section found by several queries, or
ranked first by one sub-question, rises. The raw ticket is always one of the queries, so multi-query can only add
candidates, never lose the single-query ones it would have ranked highly. The fused top ``depth`` sections then go
to the Week-6 LLM reranker (graded against the original ticket) and the top ``k_grade`` reach the evidence grader.
"""

from __future__ import annotations

from rag_cskh.retrieval import RRF_K

PER_QUERY = 20


def fuse_sections(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking, start=1):
            score[c] = score.get(c, 0.0) + 1.0 / (k + rank)
    first_seen = {c: i for i, c in enumerate(c for r in rankings for c in r)}
    return sorted(score, key=lambda c: (-score[c], first_seen[c]))


def multi_query_sections(retriever, queries: list[str], depth: int = 20, per_query: int = PER_QUERY) -> list[str]:
    return fuse_sections([retriever.search_sections(q, per_query) for q in queries])[:depth]
