"""Format checks: the document template and the example tickets follow the agreed formats."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT_MATTER = {"doc_id", "title", "category", "status", "effective_date"}
DECISION = {"normal": "answer", "missing_info": "clarify", "no_evidence": "escalate", "conflict": "escalate"}


def parse_front_matter(text: str) -> dict:
    block = text.split("---")[1]
    return dict(line.split(":", 1) for line in block.strip().splitlines())


def test_template_has_front_matter_and_section_ids():
    text = (ROOT / "data" / "kb" / "_TEMPLATE.md").read_text(encoding="utf-8")
    meta = {k.strip(): v.strip() for k, v in parse_front_matter(text).items()}
    assert FRONT_MATTER <= set(meta)
    assert meta["status"] in {"active", "expired"}
    headings = re.findall(r"^## .+$", text, flags=re.MULTILINE)
    assert headings and all(re.search(r"\{#[a-z0-9-]+\}$", h) for h in headings)


def test_example_tickets_match_schema():
    schema = json.loads((ROOT / "data" / "eval" / "ticket_schema.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (ROOT / "data" / "eval" / "examples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert {r["type"] for r in rows} == set(DECISION)
    for r in rows:
        assert set(r) == set(schema["required"])
        assert re.fullmatch(schema["properties"]["id"]["pattern"], r["id"])
        assert r["expected_decision"] == DECISION[r["type"]]
        if r["type"] in {"normal", "conflict"}:
            assert r["gold_sections"]
        if r["type"] == "no_evidence":
            assert not r["gold_sections"]   # missing_info may cite the section that states the required information
        assert bool(r["missing_slots"]) == (r["type"] == "missing_info")
        if r["type"] == "conflict":
            assert len({s.split("#")[0] for s in r["gold_sections"]}) == 2   # two different documents
