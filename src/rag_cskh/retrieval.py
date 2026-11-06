"""Retrievers over KB chunks: BM25, dense (embeddings) and hybrid (Reciprocal Rank Fusion).

All retrievers return chunk indices ranked by score. ``to_sections`` turns a chunk ranking into a ranking of
citations (``doc_id#section-id``): the first chunk of a section decides the section's rank, so results are
comparable across chunk granularities and with the gold labels.

Validity filter: with ``active_on=date`` only chunks of documents usable on that date are returned
(status active, inside effective/expiry dates). The filter is metadata, not ranking, so it is applied
identically to every method.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np

from rag_cskh.bm25 import BM25
from rag_cskh.chunking import Chunk
from rag_cskh.embeddings import Embedder
from rag_cskh.text import tokens as default_tokens

RRF_K = 60
CANDIDATES = 50


def chunk_is_active(chunk: Chunk, on: date) -> bool:
    if chunk.status != "active" or date.fromisoformat(chunk.effective_date) > on:
        return False
    return chunk.expiry_date is None or on <= date.fromisoformat(chunk.expiry_date)


class Retriever:
    name = "base"

    def __init__(self, chunks: list[Chunk], active_on: date | None = None):
        self.chunks = chunks
        self.allowed = np.array([active_on is None or chunk_is_active(c, active_on) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, k: int = CANDIDATES) -> list[int]:
        s = self.scores(query)
        order = [int(i) for i in np.argsort(-s, kind="stable") if self.allowed[i] and np.isfinite(s[i])]
        return order[:k]

    def search_sections(self, query: str, k: int = 20) -> list[str]:
        return to_sections(self.chunks, self.search(query, max(k * 3, CANDIDATES)))[:k]


class BM25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk], tokenizer: Callable[[str], list[str]] = default_tokens,
                 active_on: date | None = None, name: str = "bm25"):
        super().__init__(chunks, active_on)
        self.name = name
        self.tokenizer = tokenizer
        self.bm25 = BM25([tokenizer(c.text) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        s = np.array(self.bm25.scores(self.tokenizer(query)), dtype=np.float64)
        s[s <= 0] = -np.inf  # chunks sharing no term with the query are not retrieved
        return s


class DenseRetriever(Retriever):
    def __init__(self, chunks: list[Chunk], embedder: Embedder, active_on: date | None = None, name: str = "vector"):
        super().__init__(chunks, active_on)
        self.name = name
        self.embedder = embedder
        self.matrix = embedder.embed([c.text for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.embed([query])[0]
        return self.matrix @ q


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda d: (-fused[d], d))


class HybridRetriever(Retriever):
    def __init__(self, retrievers: list[Retriever], k: int = RRF_K, depth: int = CANDIDATES, name: str = "hybrid"):
        super().__init__(retrievers[0].chunks)
        self.allowed = np.logical_and.reduce([r.allowed for r in retrievers])
        self.name = name
        self.retrievers, self.k, self.depth = retrievers, k, depth

    def search(self, query: str, k: int = CANDIDATES) -> list[int]:
        fused = rrf([r.search(query, self.depth) for r in self.retrievers], self.k)
        return [i for i in fused if self.allowed[i]][:k]


def to_sections(chunks: list[Chunk], ranking: list[int]) -> list[str]:
    return list(dict.fromkeys(chunks[i].citation for i in ranking))
