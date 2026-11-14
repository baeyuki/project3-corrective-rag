"""Baseline RAG pipelines (Week 6): retrieve -> (rerank) -> answer with the shared prompt.

- Vector RAG: dense retrieval (text-embedding-3-small, cosine), top ``k`` sections -> answer. The usual "naive RAG".
- Hybrid RAG: folded BM25 + dense fused by RRF, top ``depth`` sections -> LLM reranker -> top ``k`` -> answer.
  Variant hybrid-rag-ce: the same with the local cross-encoder (``cross_encoder.py``) instead of the LLM reranker.
Both use the validity filter (expired documents never reach the prompt), 120-word chunks collapsed to sections,
the same ``k`` and the same answer prompt. Neither grades evidence, rewrites the query or checks citations: those
are the Corrective-RAG steps of Weeks 7-8.
"""

from __future__ import annotations

import time

from rag_cskh.answer import ANSWER_K, generate
from rag_cskh.chunking import chunk_corpus
from rag_cskh.evidence import SectionStore
from rag_cskh.rerank import RERANK_DEPTH, LLMReranker
from rag_cskh.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from rag_cskh.tickets import SNAPSHOT


class RAGPipeline:
    def __init__(self, name: str, retriever: Retriever, store: SectionStore, llm, reranker=None,
                 k: int = ANSWER_K, depth: int = RERANK_DEPTH):
        self.name, self.retriever, self.store, self.llm = name, retriever, store, llm
        self.reranker, self.k, self.depth = reranker, k, depth

    def retrieve(self, ticket: str) -> dict:
        start = time.perf_counter()
        first = self.retriever.search_sections(ticket, self.depth if self.reranker else self.k)
        retrieval_s = time.perf_counter() - start
        grades = {}
        ranked = first
        if self.reranker:
            ranked, grades = self.reranker.rerank(ticket, first)
        return {"first_stage": first, "ranked": ranked, "evidence": ranked[:self.k], "grades": grades,
                "retrieval_s": round(retrieval_s, 3)}

    def run(self, ticket: str) -> dict:
        r = self.retrieve(ticket)
        return {**r, "answer": generate(self.llm, ticket, r["evidence"], self.store)}


def build_baselines(docs, embedder, llm, rerank_llm=None, rerank: bool = True, max_words: int = 120) -> dict[str, RAGPipeline]:
    chunks = chunk_corpus(docs, max_words)
    store = SectionStore(docs)
    bm25 = BM25Retriever(chunks, active_on=SNAPSHOT)
    vector = DenseRetriever(chunks, embedder, SNAPSHOT)
    hybrid = HybridRetriever([bm25, vector])
    reranker = LLMReranker(rerank_llm or llm, store) if rerank else None
    return {
        "vector-rag": RAGPipeline("vector-rag", vector, store, llm),
        "hybrid-rag": RAGPipeline("hybrid-rag", hybrid, store, llm, reranker),
    }
