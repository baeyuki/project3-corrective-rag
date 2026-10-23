"""Assign every ticket to the dev or test split (in place in data/eval/tickets.jsonl).

    python scripts/split_tickets.py                 # 30% dev per ticket type, seed 13
    python scripts/split_tickets.py --check         # only verify that the stored split is reproducible

Design:
- Stratified by ticket type, so dev and test have the same type mix.
- Grouped, so tickets that share their evidence never straddle the split: conflict tickets are grouped by
  conflict_id (the test split contains conflicts never seen while tuning), other tickets by their first gold
  section; no_evidence tickets are independent.
- Deterministic: groups are ordered by a SHA-256 hash of seed + group key, then added to dev until the dev
  share reaches the target. Re-running gives the same split; changing the seed gives another valid split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_cskh.tickets import TICKETS_PATH, load_tickets  # noqa: E402

DEV_SHARE = 0.30
SEED = 13


def group_key(ticket: dict) -> str:
    if ticket["type"] == "conflict":
        return "conflict:" + ticket["conflict_id"]
    if ticket["gold_sections"]:
        return "section:" + ticket["gold_sections"][0]
    return "ticket:" + ticket["id"]


def assign(tickets: list[dict], dev_share: float = DEV_SHARE, seed: int = SEED) -> dict[str, str]:
    split: dict[str, str] = {}
    by_type: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in tickets:
        by_type[t["type"]][group_key(t)].append(t["id"])
    for groups in by_type.values():
        total = sum(len(ids) for ids in groups.values())
        target = round(dev_share * total)
        order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
        dev = 0
        for g in order:
            name = "dev" if dev < target and dev + len(groups[g]) <= target + 1 else "test"
            dev += len(groups[g]) if name == "dev" else 0
            for tid in groups[g]:
                split[tid] = name
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tickets = load_tickets()
    split = assign(tickets, seed=args.seed)
    if args.check:
        wrong = [t["id"] for t in tickets if t.get("split") != split[t["id"]]]
        print("split reproducible" if not wrong else f"split differs for {wrong}")
        sys.exit(1 if wrong else 0)
    for t in tickets:
        t["split"] = split[t["id"]]
    TICKETS_PATH.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tickets), encoding="utf-8")
    table = Counter((t["type"], t["split"]) for t in tickets)
    print("| Type | dev | test |\n|---|---:|---:|")
    for kind in ("normal", "missing_info", "no_evidence", "conflict"):
        print(f"| {kind} | {table[(kind, 'dev')]} | {table[(kind, 'test')]} |")
    print(f"| total | {sum(v for (k, s), v in table.items() if s == 'dev')} | {sum(v for (k, s), v in table.items() if s == 'test')} |")
    dev_conflicts = sorted({t['conflict_id'] for t in tickets if t['type'] == 'conflict' and t['split'] == 'dev'})
    print(f"conflicts in dev: {dev_conflicts}")


if __name__ == "__main__":
    main()
