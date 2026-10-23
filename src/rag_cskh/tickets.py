"""Labelled support tickets: loading and label validation.

The rules implemented here are the machine-checkable part of docs/week4_label_guide.md. They are used by
``scripts/validate_tickets.py`` (report) and ``tests/test_tickets.py`` (must pass before tickets are kept).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rag_cskh.corpus import ROOT, Document, load_corpus

TICKETS_PATH = ROOT / "data" / "eval" / "tickets.jsonl"
CONFLICTS_PATH = ROOT / "data" / "eval" / "kb_conflicts.json"
SNAPSHOT = date(2026, 10, 22)

DECISION = {"normal": "answer", "missing_info": "clarify", "no_evidence": "escalate", "conflict": "escalate"}
PREFIX = {"normal": "N", "missing_info": "M", "no_evidence": "E", "conflict": "K"}
TAGS = {
    "no_diacritics", "multi_doc", "catalog", "manufacturer", "simulated", "arithmetic", "date_reasoning",
    "near_miss", "expired_trap", "safety", "email",
}
DERIVED_TAGS = {"multi_doc", "catalog", "manufacturer", "simulated"}
SPLITS = {"dev", "test"}

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def load_tickets(path: Path = TICKETS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_conflicts(path: Path = CONFLICTS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number_forms(text: str) -> set[str]:
    """All numbers in ``text`` in comparable forms: 1.400.000 -> {1.400.000, 1400000}; 6,28 -> {6,28, 6.28}."""
    forms = set()
    for n in _NUMBER.findall(text):
        forms |= {n, n.replace(".", ""), n.replace(",", "."), n.replace(".", "").replace(",", ".")}
    return forms


def has_diacritics(text: str) -> bool:
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", text)) or "đ" in text.lower()


def derived_tags(ticket: dict, docs: dict[str, Document]) -> set[str]:
    gold_docs = {g.split("#")[0] for g in ticket["gold_sections"]}
    tags = set()
    if len(gold_docs) >= 2:
        tags.add("multi_doc")
    if any(d.startswith("SP-") for d in gold_docs):
        tags.add("catalog")
    if any(d.startswith("NSX-") for d in gold_docs):
        tags.add("manufacturer")
    if any(docs[d].meta["provenance"] == "simulated" for d in gold_docs if d in docs):
        tags.add("simulated")
    return tags


@dataclass
class Report:
    errors: list[str]
    counts: Counter
    tags: Counter

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(tickets: list[dict], docs: list[Document] | None = None, conflicts: dict | None = None) -> Report:
    docs = docs if docs is not None else load_corpus()
    conflicts = conflicts if conflicts is not None else load_conflicts()
    by_id = {d.doc_id: d for d in docs}
    sections = {s.citation: s for d in docs for s in d.sections}
    pairs = {c["id"]: c for c in conflicts["conflicts"]}
    errors: list[str] = []

    def err(t: dict, msg: str) -> None:
        errors.append(f"{t.get('id', '?')}: {msg}")

    seen_ids, seen_text = set(), {}
    for t in tickets:
        kind = t.get("type")
        if kind not in DECISION:
            err(t, f"unknown type {kind}")
            continue
        if t["expected_decision"] != DECISION[kind]:
            err(t, "decision does not match type")
        if not re.fullmatch(PREFIX[kind] + r"\d{3}", t["id"]):
            err(t, "id prefix/format")
        if t["id"] in seen_ids:
            err(t, "duplicate id")
        seen_ids.add(t["id"])
        key = re.sub(r"\W+", " ", t["ticket"].lower()).strip()
        if key in seen_text:
            err(t, f"duplicate ticket text of {seen_text[key]}")
        seen_text[key] = t["id"]

        # gold evidence
        gold = t["gold_sections"]
        for g in gold:
            if g not in sections:
                err(t, f"unknown section {g}")
            elif not by_id[g.split('#')[0]].is_active(SNAPSHOT):
                err(t, f"gold section in a document not usable on {SNAPSHOT}: {g}")
        if len(set(gold)) != len(gold):
            err(t, "repeated gold section")
        if kind == "no_evidence" and gold:
            err(t, "no_evidence must have no gold sections")
        if kind in {"normal", "conflict", "missing_info"} and not gold:
            err(t, "gold sections required")

        # type-specific fields
        if kind == "normal":
            if not t["key_facts"]:
                err(t, "normal needs key_facts")
            gold_numbers = set().union(*(number_forms(sections[g].body) for g in gold if g in sections)) if gold else set()
            if "arithmetic" not in t.get("tags", []):
                for fact in t["key_facts"]:
                    missing = {n for n in _NUMBER.findall(fact) if not number_forms(n) & gold_numbers}
                    if missing:
                        err(t, f"key fact numbers not in gold text: {sorted(missing)} in '{fact}'")
            for c in pairs.values():
                if set(c["sections"]) <= set(gold) and "near_miss" not in t.get("tags", []):
                    err(t, f"normal ticket cites both sides of conflict {c['id']} without the near_miss tag")
        elif t["key_facts"]:
            err(t, "key_facts only for normal tickets")
        if kind == "missing_info":
            if not 1 <= len(t["missing_slots"]) <= 3:
                err(t, "missing_info needs 1-3 missing_slots")
        elif t["missing_slots"]:
            err(t, "missing_slots only for missing_info tickets")
        if kind == "conflict":
            c = pairs.get(t.get("conflict_id"))
            if c is None:
                err(t, "conflict ticket needs a registered conflict_id")
            else:
                if not set(c["sections"]) <= set(gold):
                    err(t, f"gold must contain both sides of {c['id']}")
                allowed = set(c["sections"]) | set(c.get("also_conflicts_with", [])) | _same_value_sections(c)
                extra = set(gold) - allowed
                if extra:
                    err(t, f"gold sections outside conflict {c['id']}: {sorted(extra)}")
        elif "conflict_id" in t:
            err(t, "conflict_id only for conflict tickets")
        if not t["reference"].strip():
            err(t, "empty reference")
        if kind == "normal" and len(t["reference"].split()) > 120:
            err(t, "normal reference longer than 120 words (QT-MAU-PHAN-HOI-2026)")

        # tags
        tags = set(t.get("tags", []))
        if tags - TAGS:
            err(t, f"unknown tags {sorted(tags - TAGS)}")
        expected = derived_tags(t, by_id)
        if tags & DERIVED_TAGS != expected:
            err(t, f"derived tags should be {sorted(expected)}")
        if ("no_diacritics" in tags) == has_diacritics(t["ticket"]):
            err(t, "no_diacritics tag does not match the text")
        if "split" in t and t["split"] not in SPLITS:
            err(t, "invalid split")

    counts = Counter(t.get("type") for t in tickets)
    tag_counts = Counter(tag for t in tickets for tag in t.get("tags", []))
    return Report(errors, counts, tag_counts)


def _same_value_sections(conflict: dict) -> set[str]:
    """Sections allowed in conflict gold besides the registered pair: unit prices that quantify one side."""
    return set(conflict.get("supporting_sections", []))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two label sequences (used for the LLM cross-check)."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
