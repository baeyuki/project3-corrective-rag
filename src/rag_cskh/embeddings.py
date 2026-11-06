"""Text embeddings with an on-disk cache.

Every text is embedded once per model: vectors are stored in ``data/index/embeddings/<model>.jsonl`` keyed by the
SHA-256 of the text, so re-running an evaluation, changing the chunk granularity back and forth or adding tickets
only calls the API for new texts. ``data/index/`` is git-ignored and fully reproducible.

The OpenAI model is the project default (``EMBED_MODEL`` in ``.env``). ``HashingEmbedder`` is a dependency-free
stand-in used by the tests to exercise the dense and hybrid code paths offline; it is lexical, not semantic, and is
never reported as a dense-retrieval result.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_cskh.corpus import ROOT
from rag_cskh.text import tokens

CACHE_DIR = ROOT / "data" / "index" / "embeddings"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (same behaviour as scripts/check_env.py)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class CachedEmbedder:
    """Wraps a raw embedding function with the JSONL cache."""

    def __init__(self, name: str, raw_embed, cache_dir: Path = CACHE_DIR, batch_size: int = 128):
        self.name = name
        self.raw_embed = raw_embed
        self.batch_size = batch_size
        self.path = cache_dir / f"{name.replace('/', '_')}.jsonl"
        self.cache: dict[str, list[float]] = {}
        self.api_texts = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row["v"]

    def embed(self, texts: list[str]) -> np.ndarray:
        todo = list(dict.fromkeys(t for t in texts if _key(t) not in self.cache))
        if todo:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for i in range(0, len(todo), self.batch_size):
                    batch = todo[i:i + self.batch_size]
                    vectors = self.raw_embed(batch)
                    for text, vec in zip(batch, vectors):
                        self.cache[_key(text)] = list(map(float, vec))
                        f.write(json.dumps({"k": _key(text), "v": self.cache[_key(text)]}) + "\n")
            self.api_texts += len(todo)
        return normalise(np.array([self.cache[_key(t)] for t in texts], dtype=np.float32))


def openai_embedder(model: str | None = None, cache_dir: Path = CACHE_DIR) -> CachedEmbedder:
    load_env()
    model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
    client = None

    def raw(batch: list[str]) -> list[list[float]]:
        nonlocal client
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY missing: copy .env.example to .env and set it")
            from openai import OpenAI
            client = OpenAI(timeout=60)
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            except Exception:  # network or rate-limit errors: back off and retry
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return []

    return CachedEmbedder(model, raw, cache_dir)


class HashingEmbedder:
    """Offline test double: hashed bag of Vietnamese tokens (lexical, NOT semantic)."""

    name = "hashing-test"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokens(text):
                m[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalise(m)
