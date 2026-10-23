"""Knowledge-base parsing.

A KB document is a Markdown file in ``data/kb/`` (files starting with ``_`` are authoring aids, not corpus):

    ---
    doc_id: CS-DOI-TRA-2025
    title: ...
    category: policy | procedure | faq | troubleshooting | catalog | manufacturer | promotion | notice | pricelist | directory
    status: active | expired
    effective_date: 2026-01-01
    expiry_date: 2026-10-31          (optional)
    superseded_by: CS-DOI-TRA-2025   (optional, expired documents)
    variant_of: CS-GIAO-HANG-2025    (optional, planted conflicts)
    ...
    ---
    # Title
    ## Heading {#section-id}
    body

Every ``##`` section is the minimum citation unit ``doc_id#section-id``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "kb"

REQUIRED_META = ("doc_id", "title", "category", "status", "effective_date", "provenance", "source_refs")
CATEGORIES = (
    "policy", "procedure", "faq", "troubleshooting", "catalog", "manufacturer", "promotion", "notice", "pricelist", "directory",
)
STATUSES = ("active", "expired")
PROVENANCE = ("synthetic-derived", "operational-synthesis", "source-extracted", "controlled-variant", "simulated")

_HEADING = re.compile(r"^## (?P<heading>.+?) \{#(?P<id>[a-z0-9-]+)\}\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    doc_id: str
    section_id: str
    heading: str
    body: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.section_id}"


@dataclass
class Document:
    path: Path
    meta: dict[str, str]
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.meta["doc_id"]

    @property
    def category(self) -> str:
        return self.meta["category"]

    @property
    def source_refs(self) -> list[str]:
        return [r.strip() for r in self.meta.get("source_refs", "").split(",") if r.strip()]

    def is_active(self, on: date) -> bool:
        """True if the document can be used as evidence on date ``on``."""
        if self.meta.get("status") != "active":
            return False
        if date.fromisoformat(self.meta["effective_date"]) > on:
            return False
        expiry = self.meta.get("expiry_date")
        return expiry is None or on <= date.fromisoformat(expiry)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` front matter (``key: value`` lines) from the Markdown body."""
    if not text.startswith("---\n"):
        raise ValueError("document must start with front matter")
    _, block, body = text.split("---", 2)
    meta = {}
    for line in block.strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def parse_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    title_match = re.search(r"^# (.+)$", body, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else meta.get("title", path.stem)
    matches = list(_HEADING.finditer(body))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(Section(meta["doc_id"], m["id"], m["heading"].strip(), body[m.end():end].strip()))
    return Document(path=path, meta=meta, title=title, sections=sections)


def document_paths(kb_dir: Path = KB_DIR) -> list[Path]:
    return sorted(p for p in kb_dir.glob("*.md") if not p.name.startswith("_"))


def load_corpus(kb_dir: Path = KB_DIR) -> list[Document]:
    return [parse_document(p) for p in document_paths(kb_dir)]
