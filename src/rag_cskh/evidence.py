"""Section texts as they are shown to the LLM (reranker and answer prompt).

The unit is the whole ``##`` section, i.e. the citation unit: retrieval works on chunks, but the gold labels, the
reranker and the answer cite sections, and every section fits in a prompt (at most ~200 words).
"""

from __future__ import annotations

from rag_cskh.corpus import Document


class SectionStore:
    def __init__(self, docs: list[Document]):
        self.docs = {d.doc_id: d for d in docs}
        self.sections = {s.citation: (d, s) for d in docs for s in d.sections}

    def block(self, citation: str, label: str | None = None) -> str:
        """``[label] Title > Heading (metadata)`` followed by the section body."""
        d, s = self.sections[citation]
        meta = f"hiệu lực từ {d.meta['effective_date']}"
        if d.meta.get("expiry_date"):
            meta += f" đến {d.meta['expiry_date']}"
        return f"[{label or citation}] {d.title} > {s.heading} ({meta})\n{s.body}"
