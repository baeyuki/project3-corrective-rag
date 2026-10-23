"""Parsing and chunking invariants on the real corpus."""

from datetime import date

from rag_cskh.bm25 import BM25
from rag_cskh.chunking import chunk_corpus, n_words, split_blocks
from rag_cskh.corpus import load_corpus, parse_document, ROOT
from rag_cskh.text import tokens

DOCS = load_corpus()
SECTIONS = {s.citation: s for d in DOCS for s in d.sections}


def test_parser_keeps_front_matter_and_section_anchors():
    doc = parse_document(ROOT / "data" / "kb" / "CS-DOI-TRA-2024.md")
    assert doc.meta["status"] == "expired" and doc.meta["superseded_by"] == "CS-DOI-TRA-2025"
    assert [s.section_id for s in doc.sections] == ["loi-ky-thuat", "doi-do-nhu-cau", "khong-ap-dung", "hoan-tien"]
    assert "{#" not in doc.sections[0].body and not doc.sections[0].body.startswith("##")


def test_section_granularity_is_one_chunk_per_citation():
    chunks = chunk_corpus(DOCS, max_words=None)
    assert [c.citation for c in chunks] == list(SECTIONS)


def test_chunks_never_cross_sections_and_lose_no_text():
    for max_words in (120, 60):
        chunks = chunk_corpus(DOCS, max_words)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        by_citation = {}
        for c in chunks:
            assert c.citation in SECTIONS
            assert c.text.startswith(f"{c.title} > {c.heading}\n")
            by_citation.setdefault(c.citation, []).append(c)
        for citation, parts in by_citation.items():
            blocks = split_blocks(SECTIONS[citation].body)
            covered = [b for p in parts for b in split_blocks(p.body)]
            assert covered == blocks or len(parts) == 1, citation
            assert [p.part for p in parts] == list(range(1, len(parts) + 1))


def test_chunk_length_respects_limit_unless_single_block():
    for c in chunk_corpus(DOCS, 60):
        assert c.n_words <= 60 or len(split_blocks(c.body)) == 1, c.chunk_id
        assert c.n_words == n_words(c.body)


def test_long_product_sections_are_split_with_context_header():
    chunks = [c for c in chunk_corpus(DOCS, 120) if c.citation == "SP-DT-APPLE-2026#iphone-air"]
    assert len(chunks) == 2
    assert all(c.text.startswith("Danh mục điện thoại Apple iPhone đang kinh doanh > iPhone Air") for c in chunks)
    assert "eSIM" in chunks[1].body


def test_active_filter_drops_only_the_expired_policy():
    snapshot = date(2026, 10, 22)
    usable = {d.doc_id for d in DOCS if d.is_active(snapshot)}
    assert "CS-DOI-TRA-2024" not in usable and "CS-DOI-TRA-2025" in usable
    assert not next(d for d in DOCS if d.doc_id == "KM-THANG-10-2026").is_active(date(2026, 11, 1))


def test_bm25_finds_catalogue_chunks_for_unaccented_model_queries():
    chunks = chunk_corpus(DOCS, 120)
    bm25 = BM25([tokens(c.text) for c in chunks])
    top = chunks[bm25.rank(tokens("iphone air co khe sim vat ly khong"))[0]]
    assert top.citation == "SP-DT-APPLE-2026#iphone-air"
    top = chunks[bm25.rank(tokens("tem may lanh LG ghi S3-Q09JA2P5"))[0]]
    assert top.doc_id == "SP-MAY-LANH-2026"


def test_planted_conflicts_are_reachable_by_bm25():
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "probe_conflicts.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    for row in probe.probe():
        assert row["rank_a"] is not None and row["rank_a"] <= 10, row
        assert row["rank_b"] is not None and row["rank_b"] <= 10, row
