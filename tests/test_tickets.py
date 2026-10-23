"""Label checks for the Week-4 evaluation set (data/eval/tickets.jsonl)."""

import importlib.util
import json
from collections import Counter

from rag_cskh.corpus import ROOT, load_corpus
from rag_cskh.tickets import cohen_kappa, load_conflicts, load_tickets, validate

TICKETS = load_tickets()


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_label_rules_pass():
    report = validate(TICKETS)
    assert report.ok, "\n".join(report.errors)


def test_size_and_type_balance():
    counts = Counter(t["type"] for t in TICKETS)
    assert 140 <= len(TICKETS) <= 180
    assert counts["normal"] / len(TICKETS) < 0.6
    assert min(counts[k] for k in ("missing_info", "no_evidence", "conflict")) >= 20


def test_every_planted_conflict_has_tickets_and_every_ticket_matches_the_schema():
    schema = json.loads((ROOT / "data" / "eval" / "ticket_schema.json").read_text(encoding="utf-8"))
    allowed = set(schema["properties"])
    for t in TICKETS:
        assert set(schema["required"]) <= set(t) <= allowed, t["id"]
    covered = Counter(t["conflict_id"] for t in TICKETS if t["type"] == "conflict")
    assert set(covered) == {c["id"] for c in load_conflicts()["conflicts"]}
    assert min(covered.values()) >= 3


def test_hard_cases_are_represented():
    tags = Counter(tag for t in TICKETS for tag in t["tags"])
    assert tags["no_diacritics"] / len(TICKETS) >= 0.15
    for tag in ("near_miss", "expired_trap", "multi_doc", "arithmetic", "date_reasoning", "safety"):
        assert tags[tag] >= 3, tag


def test_split_is_stratified_grouped_and_reproducible():
    split = load_script("split_tickets").assign(TICKETS)
    assert all(t["split"] == split[t["id"]] for t in TICKETS)
    for kind in ("normal", "missing_info", "no_evidence", "conflict"):
        rows = [t for t in TICKETS if t["type"] == kind]
        dev = sum(t["split"] == "dev" for t in rows) / len(rows)
        assert 0.2 <= dev <= 0.4, kind
    by_group = {}
    group_key = load_script("split_tickets").group_key
    for t in TICKETS:
        assert by_group.setdefault(group_key(t), t["split"]) == t["split"], t["id"]


def test_validator_catches_typical_label_errors():
    good = next(t for t in TICKETS if t["id"] == "N040")
    bad = [
        dict(good, id="N900", ticket="x1", expected_decision="escalate"),
        dict(good, id="N901", ticket="x2", gold_sections=["CS-DOI-TRA-2024#loi-ky-thuat"], tags=[]),
        dict(good, id="N902", ticket="x3", key_facts=["99 W"]),
        dict(good, id="E900", ticket="x4", type="no_evidence", expected_decision="escalate", key_facts=[]),
    ]
    errors = validate(bad).errors
    assert any(e.startswith("N900") and "decision" in e for e in errors)
    assert any(e.startswith("N901") and "not usable" in e for e in errors)
    assert any(e.startswith("N902") and "99" in e for e in errors)
    assert any(e.startswith("E900") and "no gold" in e for e in errors)


def test_kappa_helper():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


def test_llm_cross_check_is_blind_to_reference_labels():
    check = load_script("llm_label_check")
    docs = load_corpus()
    toc = check.table_of_contents(docs)
    assert "CS-DOI-TRA-2024" not in toc            # expired documents are not offered as evidence
    candidates = check.Candidates(docs, dense=False)
    for t in TICKETS[::7]:
        cits = candidates.citations(t["ticket"])
        prompt = json.dumps(check.build_messages(t["ticket"], toc, candidates.text(cits)), ensure_ascii=False)
        assert t["reference"] not in prompt and t["id"] not in prompt
        assert "conflict_id" not in prompt and "kb_conflicts" not in prompt
        assert all(f"#{c.split('#')[1]}:" in toc and c.split("#")[0] in toc for c in cits)
