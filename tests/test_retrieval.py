"""Retrieval and metric tests (offline: the dense path uses the hashing test double, never the API)."""

import math
from datetime import date

import numpy as np

from rag_cskh.chunking import chunk_corpus
from rag_cskh.corpus import load_corpus
from rag_cskh.embeddings import CachedEmbedder, HashingEmbedder
from rag_cskh.metrics import both_sides, ndcg, recall, reciprocal_rank, ticket_metrics
from rag_cskh.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, rrf, to_sections

CHUNKS = chunk_corpus(load_corpus(), 120)
SNAPSHOT = date(2026, 10, 22)


def test_metrics_on_hand_computed_example():
    ranked = ["a", "x", "b", "y"]
    gold = {"a", "b"}
    assert recall(ranked, gold, 1) == 0.5 and recall(ranked, gold, 3) == 1.0
    assert reciprocal_rank(["x", "a"], gold) == 0.5 and reciprocal_rank(["x"], gold) == 0.0
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(ndcg(ranked, gold, 3) - expected) < 1e-12
    assert both_sides(["b2", "a"], ["a", "b1", "b2"], 2) == 1.0
    assert both_sides(["a", "z"], ["a", "b1", "b2"], 2) == 0.0
    m = ticket_metrics(ranked, ["a", "b"])
    assert m["All@3"] == 1.0 and m["All@1"] == 0.0 and m["Hit@1"] == 1.0


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert fused[0] == 1                     # ranks 1 and 2
    assert fused.index(3) < fused.index(2)   # ranks 3 and 1 beat 2 alone
    assert set(fused) == {1, 2, 3, 4}


def test_validity_filter_removes_only_the_expired_policy():
    unfiltered = BM25Retriever(CHUNKS)
    filtered = BM25Retriever(CHUNKS, active_on=SNAPSHOT)
    query = "doi do nhu cau trong bao nhieu ngay phi bao nhieu"
    assert any(c.startswith("CS-DOI-TRA-2024") for c in unfiltered.search_sections(query, 5))
    assert not any(c.startswith("CS-DOI-TRA-2024") for c in filtered.search_sections(query, 20))
    removed = {CHUNKS[i].doc_id for i in range(len(CHUNKS)) if not filtered.allowed[i]}
    assert removed == {"CS-DOI-TRA-2024"}


def test_section_ranking_deduplicates_split_chunks():
    ranking = [i for i, c in enumerate(CHUNKS) if c.citation == "SP-DT-APPLE-2026#iphone-air"]
    assert len(ranking) == 2
    assert to_sections(CHUNKS, ranking) == ["SP-DT-APPLE-2026#iphone-air"]


def test_dense_and_hybrid_paths_run_offline():
    dense = DenseRetriever(CHUNKS, HashingEmbedder(), SNAPSHOT, name="hash")
    bm25 = BM25Retriever(CHUNKS, active_on=SNAPSHOT)
    hybrid = HybridRetriever([bm25, dense])
    query = "iphone air co khe sim vat ly khong"
    for r in (dense, hybrid):
        top = r.search_sections(query, 5)
        assert "SP-DT-APPLE-2026#iphone-air" in top
        assert len(top) == len(set(top))


def test_embedding_cache_calls_the_api_once_per_text(tmp_path):
    calls = []

    def fake_api(batch):
        calls.append(list(batch))
        return [[float(len(t)), 1.0] for t in batch]

    emb = CachedEmbedder("fake-model", fake_api, cache_dir=tmp_path, batch_size=2)
    first = emb.embed(["a", "bb", "a", "ccc"])
    assert sum(len(b) for b in calls) == 3 and max(len(b) for b in calls) <= 2
    again = CachedEmbedder("fake-model", fake_api, cache_dir=tmp_path).embed(["ccc", "bb"])
    assert sum(len(b) for b in calls) == 3                       # served from the file cache
    assert np.allclose(again[0], first[3]) and np.allclose(np.linalg.norm(first, axis=1), 1)
