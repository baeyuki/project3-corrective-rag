"""Cross-encoder reranker (Week 6, local alternative to the LLM reranker).

A cross-encoder reads (ticket, section) together and outputs one relevance score; it runs locally (no API cost),
and is the standard second stage of retrieve-then-rerank pipelines. Same interface as ``LLMReranker``: the hybrid
top ``depth`` sections are re-sorted by score, ties keep the first-stage order, nothing is dropped.

Presets (``--ce-model``), all on Hugging Face:
- ``bge-m3``    BAAI/bge-reranker-v2-m3: multilingual (XLM-R / BGE-M3 backbone, 568M parameters), raw text,
                max_length 512 tokens (every KB section is at most ~200 words). Default.
- ``phoranker`` itdainb/PhoRanker: Vietnamese (PhoBERT), trained on mMARCO-vi. PhoBERT expects word-segmented
                input, so query and section are segmented with pyvi; max_length 256, so long sections are truncated.
- ``viranker``  namdp-ptit/ViRanker: Vietnamese, BGE-M3 backbone, trained on mMARCO-vi.
- any other Hugging Face id is used as is (raw text).
The section text given to the model is "document title > heading" followed by the body (no metadata line).
Scores are cached in ``data/index/ce_cache/<model>.jsonl`` keyed by SHA-256 of (query, passage), so re-runs and
report rebuilds are free. ``sentence-transformers`` (and ``pyvi`` for PhoRanker) are imported lazily: the rest of
the project, and the tests, do not need them.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from rag_cskh.corpus import ROOT
from rag_cskh.evidence import SectionStore
from rag_cskh.rerank import RERANK_DEPTH

CACHE_DIR = ROOT / "data" / "index" / "ce_cache"

PRESETS = {
    "bge-m3": {"model": "BAAI/bge-reranker-v2-m3", "max_length": 512, "segment": False},
    "phoranker": {"model": "itdainb/PhoRanker", "max_length": 256, "segment": True},
    "viranker": {"model": "namdp-ptit/ViRanker", "max_length": 512, "segment": False},
}


def resolve(name: str) -> dict:
    return PRESETS.get(name, {"model": name, "max_length": 512, "segment": False})


def passage(store: SectionStore, citation: str) -> str:
    d, s = store.sections[citation]
    return f"{d.title} > {s.heading}\n{s.body}"


class CrossEncoderReranker:
    """``predict(pairs) -> list[float]`` is the model; ``load_cross_encoder`` builds it from a preset."""

    def __init__(self, predict, store: SectionStore, model_name: str, depth: int = RERANK_DEPTH,
                 segment=None, cache_dir: Path | None = CACHE_DIR):
        self.predict, self.store, self.depth = predict, store, depth
        self.name = f"ce:{model_name}"
        self.model_name = model_name
        self.segment = segment or (lambda t: t)
        self.cache: dict[str, float] = {}
        self.pairs_scored = 0
        self._lock = threading.Lock()
        self.path = cache_dir / f"{model_name.replace('/', '_')}.jsonl" if cache_dir else None
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row["s"]

    @staticmethod
    def key(query: str, text: str) -> str:
        return hashlib.sha256(json.dumps([query, text], ensure_ascii=False).encode("utf-8")).hexdigest()

    def scores(self, query: str, texts: list[str]) -> list[float]:
        keys = [self.key(query, t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            q = self.segment(query)
            raw = self.predict([(q, self.segment(texts[i])) for i in todo])
            with self._lock:
                self.pairs_scored += len(todo)
                for i, s in zip(todo, raw):
                    self.cache[keys[i]] = float(s)
                if self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        for i in todo:
                            f.write(json.dumps({"k": keys[i], "s": self.cache[keys[i]]}) + "\n")
        return [self.cache[k] for k in keys]

    def rerank(self, ticket: str, citations: list[str]) -> tuple[list[str], dict[str, float]]:
        head, tail = citations[:self.depth], citations[self.depth:]
        if not head:
            return [], {}
        s = self.scores(ticket, [passage(self.store, c) for c in head])
        score = dict(zip(head, s))
        ranked = sorted(head, key=lambda c: (-score[c], head.index(c)))
        return ranked + tail, score


def load_cross_encoder(store: SectionStore, preset: str = "bge-m3", device: str | None = None, fp16: bool = False,
                       batch_size: int = 16, depth: int = RERANK_DEPTH) -> CrossEncoderReranker:
    cfg = resolve(preset)
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:  # pragma: no cover - depends on the local install
        raise RuntimeError("pip install -r requirements-ce.txt (sentence-transformers, torch)") from e
    model = CrossEncoder(cfg["model"], max_length=cfg["max_length"], device=device)
    if fp16 and str(model.model.device).startswith("cuda"):
        model.model.half()
    segment = None
    if cfg["segment"]:
        try:
            from pyvi import ViTokenizer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PhoRanker needs word segmentation: pip install pyvi") from e
        segment = ViTokenizer.tokenize

    def predict(pairs):
        return [float(x) for x in model.predict(pairs, batch_size=batch_size, show_progress_bar=False)]

    return CrossEncoderReranker(predict, store, cfg["model"], depth, segment)
