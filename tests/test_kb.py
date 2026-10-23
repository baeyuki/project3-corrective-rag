"""Quality checks for the knowledge-base corpus (Week 2 baseline + Week 3 extension)."""

import json
import re
from collections import Counter
from datetime import date

from rag_cskh.corpus import CATEGORIES, PROVENANCE, REQUIRED_META, ROOT, STATUSES, load_corpus

SNAPSHOT = date(2026, 10, 22)
EXPECTED_CATEGORIES = {
    "policy": 11,
    "procedure": 12,
    "faq": 10,
    "troubleshooting": 7,
    "catalog": 10,
    "manufacturer": 6,
    "promotion": 1,
    "notice": 3,
    "pricelist": 2,
    "directory": 1,
}
DOCS = load_corpus()
BY_ID = {d.doc_id: d for d in DOCS}
CITATIONS = {s.citation: s for d in DOCS for s in d.sections}


def registry_ids() -> set[str]:
    ids = set()
    for name in ("week2_sources.md", "week3_sources.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        ids |= set(re.findall(r"^### (SRC-[A-Z0-9-]+)$", text, flags=re.MULTILINE))
    return ids


def test_document_count_and_category_balance():
    assert len(DOCS) == 63
    assert Counter(d.category for d in DOCS) == EXPECTED_CATEGORIES
    assert set(EXPECTED_CATEGORIES) == set(CATEGORIES)


def test_metadata_is_complete_and_consistent():
    assert len(BY_ID) == len(DOCS)
    for d in DOCS:
        assert set(REQUIRED_META) <= set(d.meta), d.path.name
        assert d.doc_id == d.path.stem
        assert d.meta["status"] in STATUSES
        assert d.meta["provenance"] in PROVENANCE
        start = date.fromisoformat(d.meta["effective_date"])
        if "expiry_date" in d.meta:
            assert date.fromisoformat(d.meta["expiry_date"]) >= start, d.doc_id
        for key in ("superseded_by", "variant_of"):
            if key in d.meta:
                assert d.meta[key] in BY_ID, f"{d.doc_id}: {key} -> {d.meta[key]}"


def test_exactly_one_expired_policy_with_active_replacement():
    expired = [d for d in DOCS if d.meta["status"] == "expired"]
    assert [d.category for d in expired] == ["policy"]
    old = expired[0]
    new = BY_ID[old.meta["superseded_by"]]
    assert date.fromisoformat(old.meta["expiry_date"]) < date.fromisoformat(new.meta["effective_date"])
    assert new.is_active(SNAPSHOT) and not old.is_active(SNAPSHOT)
    assert {s.section_id for s in old.sections} == {s.section_id for s in new.sections}


def test_all_other_documents_are_usable_on_snapshot():
    assert sum(d.is_active(SNAPSHOT) for d in DOCS) == len(DOCS) - 1


def test_source_references_resolve_and_registry_is_used():
    registered = registry_ids()
    used = set()
    for d in DOCS:
        refs = set(d.source_refs)
        assert refs, d.doc_id
        assert refs <= registered, f"{d.doc_id}: unknown sources {sorted(refs - registered)}"
        used |= refs
    assert used == registered, f"unused registry sources: {sorted(registered - used)}"


def test_catalogue_and_manufacturer_documents_are_source_extracted():
    for d in DOCS:
        if d.category in {"catalog", "manufacturer"}:
            assert d.meta["provenance"] == "source-extracted", d.doc_id
            assert d.meta.get("brand"), d.doc_id
        if d.meta["provenance"] == "source-extracted":
            assert d.category in {"catalog", "manufacturer"}, d.doc_id
    for d in DOCS:
        if d.category == "catalog":
            assert d.sections[0].section_id == "pham-vi", d.doc_id


def test_simulated_documents_declare_the_simulation_spec():
    simulated = [d for d in DOCS if d.meta["provenance"] == "simulated"]
    assert len(simulated) == 13
    for d in simulated:
        assert "SRC-SIM-RULES" in d.source_refs, d.doc_id
        assert d.meta.get("owner"), d.doc_id
        assert d.category not in {"catalog", "manufacturer"}, d.doc_id


def test_sections_have_stable_unique_ids_and_content():
    seen = set()
    for d in DOCS:
        assert len(d.sections) >= 3, d.doc_id
        for s in d.sections:
            assert re.fullmatch(r"[a-z0-9-]+", s.section_id)
            assert s.citation not in seen
            seen.add(s.citation)
            assert len(s.body) >= 80, s.citation
    assert len(seen) == len(CITATIONS)


def test_conflict_registry_points_to_active_documents_in_two_files():
    registry = json.loads((ROOT / "data" / "eval" / "kb_conflicts.json").read_text(encoding="utf-8"))
    conflicts = registry["conflicts"]
    assert len(conflicts) >= 5
    planted = set()
    for c in conflicts:
        a, b = c["sections"]
        assert a in CITATIONS and b in CITATIONS, c["id"]
        doc_a, doc_b = BY_ID[a.split("#")[0]], BY_ID[b.split("#")[0]]
        assert doc_a.doc_id != doc_b.doc_id
        assert doc_a.is_active(SNAPSHOT) and doc_b.is_active(SNAPSHOT), c["id"]
        assert doc_a.meta.get("variant_of") == doc_b.doc_id, c["id"]
        assert len(c["rules"]) == 2 and c["rules"][0] != c["rules"][1]
        planted.add(doc_a.doc_id)
    variants = {d.doc_id for d in DOCS if d.meta["provenance"] == "controlled-variant" and d.meta["status"] == "active"}
    assert planted == variants
    for s in registry["superseded"]:
        assert BY_ID[s["expired"]].meta["superseded_by"] == s["active"]
    for n in registry["compatible"]:
        assert all(ref in CITATIONS for ref in n["sections"]), n["id"]


def test_conflicting_values_really_appear_in_the_documents():
    """The registry must describe the text, not an intention: every number in a rule appears in its section."""
    registry = json.loads((ROOT / "data" / "eval" / "kb_conflicts.json").read_text(encoding="utf-8"))
    for c in registry["conflicts"]:
        for ref, rule in zip(c["sections"], c["rules"]):
            body = CITATIONS[ref].body
            for number in re.findall(r"\d[\d.]*", rule):
                assert number in body, f"{c['id']}: {number} not in {ref}"


def test_kb_is_utf8_vietnamese_without_replacement_characters():
    corpus = "\n".join(d.path.read_text(encoding="utf-8") for d in DOCS)
    assert "�" not in corpus
    assert all(term in corpus.lower() for term in ("bảo hành", "đổi trả", "giao hàng", "thanh toán", "thông số"))
