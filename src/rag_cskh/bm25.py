"""Okapi BM25 (Robertson & Zaragoza, 2009)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.lengths = [len(d) for d in docs]
        self.avgdl = sum(self.lengths) / max(1, self.n)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> list[float]:
        s = [0.0] * self.n
        for term in dict.fromkeys(query):
            for i, tf in self.postings.get(term, []):
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return s

    def rank(self, query: list[str]) -> list[int]:
        s = self.scores(query)
        return sorted((i for i in range(self.n) if s[i] > 0), key=lambda i: -s[i])
