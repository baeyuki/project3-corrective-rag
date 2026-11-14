"""Chat-model calls with Structured Outputs and an on-disk cache.

Every call is keyed by SHA-256 of (model, messages, schema) and stored in ``data/index/llm_cache/<model>.jsonl``
with its token usage and latency, so re-running an evaluation or rebuilding a report never pays twice, and the
reported cost and latency are those of the original call. Temperature is 0.

``FakeLLM`` is the offline test double: a Python function decides the answer, the cache and accounting code paths
are the same.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag_cskh.corpus import ROOT
from rag_cskh.embeddings import load_env

CACHE_DIR = ROOT / "data" / "index" / "llm_cache"

# USD per 1M tokens (input, output), OpenAI list prices; used only for the cost estimate in reports.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40)}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, row: dict, cached: bool) -> None:
        with self.lock:
            self.calls += 1
            self.cached += cached
            self.prompt_tokens += row["prompt_tokens"]
            self.completion_tokens += row["completion_tokens"]
            self.latency_s += row["latency_s"]

    def cost(self, model: str) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1e6


def call_key(model: str, messages: list[dict], schema: dict) -> str:
    payload = json.dumps([model, messages, schema], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedLLM:
    """``chat_json(messages, schema, name) -> (parsed_json, record)``; ``raw_call`` does the actual API request."""

    def __init__(self, model: str, raw_call, cache_dir: Path = CACHE_DIR):
        self.model = model
        self.raw_call = raw_call
        self.path = cache_dir / f"{model.replace('/', '_')}.jsonl"
        self.cache: dict[str, dict] = {}
        self.usage = Usage()
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row

    def chat_json(self, messages: list[dict], schema: dict, name: str = "output") -> tuple[dict, dict]:
        key = call_key(self.model, messages, schema)
        row = self.cache.get(key)
        cached = row is not None
        if row is None:
            start = time.perf_counter()
            content, prompt_tokens, completion_tokens = self.raw_call(messages, schema, name)
            row = {"k": key, "out": json.loads(content), "prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens, "latency_s": round(time.perf_counter() - start, 3)}
            with self._lock:
                self.cache[key] = row
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.usage.add(row, cached)
        return row["out"], row


def openai_llm(model: str | None = None, env_var: str = "LLM_MODEL", cache_dir: Path = CACHE_DIR) -> CachedLLM:
    load_env()
    model = model or os.getenv(env_var, "gpt-4o-mini")
    client = None

    def raw(messages, schema, name):
        nonlocal client
        if client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY missing: copy .env.example to .env and set it")
            from openai import OpenAI
            client = OpenAI(timeout=120)
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0, messages=messages,
                    response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}})
                return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens
            except Exception:  # network or rate-limit errors: back off and retry
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    return CachedLLM(model, raw, cache_dir)


class FakeLLM(CachedLLM):
    """Offline test double: ``respond(messages, schema) -> dict``; token counts are word counts."""

    def __init__(self, respond, cache_dir: Path, model: str = "fake-model"):
        def raw(messages, schema, name):
            out = respond(messages, schema)
            words = sum(len(m["content"].split()) for m in messages)
            return json.dumps(out, ensure_ascii=False), words, len(json.dumps(out).split())
        super().__init__(model, raw, cache_dir)
