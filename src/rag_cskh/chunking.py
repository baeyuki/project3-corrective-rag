"""Section-aware chunking.

Design (Week 3):
- A chunk never crosses a ``##`` section, so every chunk maps to exactly one citation ``doc_id#section-id``.
- A section whose body fits in ``max_words`` is one chunk. A longer section is split on block boundaries
  (paragraphs and bullet items) and packed greedily up to ``max_words``; ``overlap_blocks`` trailing blocks are
  repeated at the start of the next part. A single block longer than ``max_words`` is kept whole (never cut
  mid-sentence).
- Each chunk's ``text`` starts with a context line "document title > section heading" so that BM25 and
  embeddings see which product or policy a bullet list belongs to; ``body`` keeps the raw section text.
- Lengths are measured in whitespace words. In Vietnamese a written word is one syllable, so this equals the
  syllable count used by the tokeniser in ``text.py`` (before bigrams).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rag_cskh.corpus import Document

DEFAULT_MAX_WORDS = 120

_BLOCK_START = re.compile(r"^(?:[-*] |\d+\. )")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    citation: str
    doc_id: str
    section_id: str
    part: int
    n_parts: int
    title: str
    heading: str
    category: str
    status: str
    effective_date: str
    expiry_date: str | None
    superseded_by: str | None
    variant_of: str | None
    brand: str | None
    provenance: str
    body: str
    text: str
    n_words: int

    def to_dict(self) -> dict:
        return asdict(self)


def n_words(text: str) -> int:
    return len(text.split())


def split_blocks(body: str) -> list[str]:
    """Paragraphs and list items; continuation lines stay with their item."""
    blocks: list[list[str]] = []
    for line in body.splitlines():
        if not line.strip():
            if blocks and blocks[-1]:
                blocks.append([])
            continue
        if _BLOCK_START.match(line) or not blocks or not blocks[-1]:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return ["\n".join(b) for b in blocks if b]


def pack_blocks(blocks: list[str], max_words: int, overlap_blocks: int = 0) -> list[list[str]]:
    """Greedy packing towards a balanced target (total / number of parts) so the last part is not a stub."""
    total = sum(n_words(b) for b in blocks)
    n_parts = -(-total // max_words)
    target = min(max_words, -(-total // n_parts))
    parts: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        size = n_words("\n".join(current + [block]))
        if current and (size > max_words or n_words("\n".join(current)) >= target):
            parts.append(current)
            current = current[-overlap_blocks:] if overlap_blocks else []
            if current and n_words("\n".join(current + [block])) > max_words:
                current = []
        current.append(block)
    if current:
        parts.append(current)
    return parts


def chunk_document(doc: Document, max_words: int | None = DEFAULT_MAX_WORDS, overlap_blocks: int = 0) -> list[Chunk]:
    """``max_words=None`` gives one chunk per section."""
    meta = doc.meta
    chunks = []
    for section in doc.sections:
        if max_words is None or n_words(section.body) <= max_words:
            parts = [[section.body]]
        else:
            parts = pack_blocks(split_blocks(section.body), max_words, overlap_blocks)
        for i, blocks in enumerate(parts, start=1):
            body = "\n".join(blocks).strip()
            suffix = "" if len(parts) == 1 else f"~{i}"
            chunks.append(
                Chunk(
                    chunk_id=section.citation + suffix,
                    citation=section.citation,
                    doc_id=doc.doc_id,
                    section_id=section.section_id,
                    part=i,
                    n_parts=len(parts),
                    title=doc.title,
                    heading=section.heading,
                    category=meta["category"],
                    status=meta["status"],
                    effective_date=meta["effective_date"],
                    expiry_date=meta.get("expiry_date"),
                    superseded_by=meta.get("superseded_by"),
                    variant_of=meta.get("variant_of"),
                    brand=meta.get("brand"),
                    provenance=meta["provenance"],
                    body=body,
                    text=f"{doc.title} > {section.heading}\n{body}",
                    n_words=n_words(body),
                )
            )
    return chunks


def chunk_corpus(docs: list[Document], max_words: int | None = DEFAULT_MAX_WORDS, overlap_blocks: int = 0) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, max_words, overlap_blocks)]
