"""Quality checks for the Week-2 knowledge-base corpus."""

from collections import Counter
from datetime import date
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
REQUIRED = {"doc_id", "title", "category", "status", "effective_date", "provenance", "source_refs"}
PROVENANCE = {"synthetic-derived", "operational-synthesis"}
EXPECTED_CATEGORIES = {
    "policy": 7,
    "procedure": 6,
    "faq": 5,
    "troubleshooting": 7,
}


def documents() -> list[Path]:
    return sorted(p for p in KB.glob("*.md") if not p.name.startswith("_"))


def parse_front_matter(text: str) -> dict[str, str]:
    assert text.startswith("---\n")
    block = text.split("---", 2)[1]
    return {key.strip(): value.strip() for key, value in (line.split(":", 1) for line in block.strip().splitlines())}


def test_week2_document_count_and_category_balance():
    docs = documents()
    assert len(docs) == 25
    categories = Counter(parse_front_matter(p.read_text(encoding="utf-8"))["category"] for p in docs)
    assert categories == EXPECTED_CATEGORIES


def test_kb_metadata_is_complete_and_unique():
    seen_ids = set()
    for path in documents():
        text = path.read_text(encoding="utf-8")
        meta = parse_front_matter(text)
        assert REQUIRED <= set(meta), path.name
        assert meta["doc_id"] == path.stem
        assert meta["doc_id"] not in seen_ids
        assert meta["status"] == "active"
        date.fromisoformat(meta["effective_date"])
        assert meta["provenance"] in PROVENANCE
        assert meta["source_refs"]
        seen_ids.add(meta["doc_id"])


def test_kb_source_references_resolve_to_registry():
    registry = (ROOT / "docs" / "week2_sources.md").read_text(encoding="utf-8")
    registered = set(re.findall(r"^### (SRC-[A-Z0-9-]+)$", registry, flags=re.MULTILINE))
    assert registered
    used = set()
    for path in documents():
        meta = parse_front_matter(path.read_text(encoding="utf-8"))
        refs = {ref.strip() for ref in meta["source_refs"].split(",")}
        assert refs <= registered, f"{path.name}: unknown sources {sorted(refs - registered)}"
        used |= refs
    assert used == registered, f"unused registry sources: {sorted(registered - used)}"


def test_kb_sections_have_stable_unique_ids_and_content():
    references = set()
    for path in documents():
        text = path.read_text(encoding="utf-8")
        meta = parse_front_matter(text)
        headings = re.findall(r"^## (.+) \{#([a-z0-9-]+)\}$", text, flags=re.MULTILINE)
        assert len(headings) >= 3, path.name
        for index, (_, section_id) in enumerate(headings):
            ref = f'{meta["doc_id"]}#{section_id}'
            assert ref not in references
            references.add(ref)
            start = text.index(f"{{#{section_id}}}") + len(section_id) + 3
            end = text.find("\n## ", start)
            body = text[start:end if end >= 0 else None].strip()
            assert len(body) >= 80, ref


def test_kb_is_utf8_vietnamese_without_replacement_characters():
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in documents())
    assert "�" not in corpus
    assert all(term in corpus.lower() for term in ("bảo hành", "đổi trả", "giao hàng", "thanh toán"))
