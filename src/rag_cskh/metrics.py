"""Retrieval metrics at citation (section) level with binary relevance.

For a ranked list of citations and a set of gold citations:
- Hit@K      1 if any gold section is in the top K.
- Recall@K   fraction of gold sections in the top K (multi-document tickets need all of them).
- All@K      1 if every gold section is in the top K.
- MRR        1 / rank of the first gold section (0 if none in the ranking).
- nDCG@K     binary-gain DCG normalised by the ideal DCG for min(|gold|, K) relevant items (Järvelin & Kekäläinen 2002).
- Both@K     conflict tickets only: side A (first gold section) and at least one section of side B are in the top K;
             a grader cannot detect a conflict it never sees.
Bootstrap confidence intervals resample tickets with replacement; paired differences resample the same tickets
for both methods.
"""

from __future__ import annotations

import math
import random
from statistics import mean

KS = (1, 3, 5, 10, 20)


def hit(ranked: list[str], gold: set[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & gold))


def recall(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def all_found(ranked: list[str], gold: set[str], k: int) -> float:
    return float(gold <= set(ranked[:k]))


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, c in enumerate(ranked, start=1):
        if c in gold:
            return 1.0 / i
    return 0.0


def ndcg(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], start=1) if c in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def both_sides(ranked: list[str], gold: list[str], k: int) -> float:
    top = set(ranked[:k])
    return float(gold[0] in top and bool(top & set(gold[1:])))


def ticket_metrics(ranked: list[str], gold_list: list[str], is_conflict: bool = False) -> dict[str, float]:
    gold = set(gold_list)
    m = {"MRR": reciprocal_rank(ranked, gold)}
    for k in KS:
        m[f"Hit@{k}"] = hit(ranked, gold, k)
        m[f"Recall@{k}"] = recall(ranked, gold, k)
        m[f"All@{k}"] = all_found(ranked, gold, k)
        m[f"nDCG@{k}"] = ndcg(ranked, gold, k)
        if is_conflict:
            m[f"Both@{k}"] = both_sides(ranked, gold_list, k)
    return m


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {k: mean(r[k] for r in rows) for k in keys}


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_diff_ci(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) with a 95% bootstrap interval over the same tickets."""
    diffs = [y - x for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, n, seed)
    return mean(diffs), lo, hi
