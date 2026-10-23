# Week 4 - Label guide for support tickets

This guide is the single reference for labelling `data/eval/tickets.jsonl`. Every label must be reproducible from the ticket text, this guide and the knowledge base (KB) alone; this is checked by an LLM cross-check (section 6).

## 1. Fixed context

- **Date of every ticket:** Thursday 2026-10-22 (the corpus snapshot). "Hôm nay", "tuần trước" and the validity of documents are resolved against this date.
- **Evidence:** only documents in `data/kb/` that are usable on that date (`status: active`, `effective_date` ≤ date ≤ `expiry_date`). `CS-DOI-TRA-2024` is expired and is never gold evidence.
- **Role of the system:** agent assist. The output is a draft reply for a human agent, so "escalate" means "do not commit; route to the shift lead or owning department".
- **No outside knowledge.** A fact that is true in the world but not stated in the KB counts as missing from the KB.

## 2. Decision procedure

Apply the questions in order; the first "yes" decides the label.

1. **Do two active documents state different values for the case described in the ticket?** → `conflict`, decision `escalate`. The case must fall inside the `applies_when` of a registered conflict in `data/eval/kb_conflicts.json` (for example a promotion item ordered in October for C01, an order between VND 300,000 and 499,999 within 10 km for C02). If the ticket's case lies outside that range, the two documents agree and the ticket is not a conflict (see near-misses, section 4).
2. **Does no active document contain the information needed to answer?** → `no_evidence`, decision `escalate`. This includes values a catalogue marks as unpublished, live data (current price, stock), services the KB lists as out of scope (rental, trade-in, repairing products bought elsewhere) and products not in the catalogue.
3. **Does the KB answer depend on customer-specific information that the ticket does not give, and does a document say this information is needed?** → `missing_info`, decision `clarify`. Typical cases: the order code before any order action, the exact model when variants differ (Galaxy S26 vs S26+ vs S26 Ultra), the Lenovo machine-type model, origin and activation for Panasonic warranty, the delivery date when a deadline must be checked.
4. Otherwise → `normal`, decision `answer`.

Tie-breakers:

- A general question about a rule ("đổi trả trong mấy ngày?") is `normal` even if acting on it later needs the order code; it becomes `missing_info` only when the customer asks the agent to act or to judge their own case and a required item is missing.
- If the ticket gives enough detail to decide (dates, amount, distance, model), it is `normal` even if the agent would still ask for the order code to execute.
- Safety signs (smoke, burning smell, sparks, swollen battery, water near electricity) are labelled by the same procedure, usually `normal`; the reference must start with the safety instruction and the manager escalation from `QT-SU-CO-AN-TOAN-2026`, whatever else the customer asked.
- **No precedence between documents.** The KB defines none: a newer notice, a promotion or a more specific FAQ does not override a policy unless it says so explicitly. Two active documents that give different results for the customer's case are a `conflict` (`QT-SU-DUNG-TAI-LIEU-2026#tai-lieu-mau-thuan`), even when one of them looks more recent or more specific.
- **Stated absence is not evidence.** A catalogue line saying a value is unpublished, or a procedure saying a value must be looked up in the live sales system (price, stock), does not answer the question: `no_evidence`.
- **Variants.** For an informational question whose answer differs by variant, answer all variants (`normal`) unless the KB itself says to ask back first (the "Lưu ý khi tư vấn" section of a catalogue); then `missing_info`.

## 3. Fields

| Field | Rule |
|---|---|
| `id` | `N`, `M`, `E`, `K` + three digits, unique. |
| `type`, `expected_decision` | normal → answer, missing_info → clarify, no_evidence → escalate, conflict → escalate. |
| `ticket` | Customer text as typed: chat or email style, with or without diacritics, abbreviations ("ko", "dc", "tr") allowed. One main intent per ticket. |
| `gold_sections` | `normal`: every section needed for a complete answer (the minimum set). `conflict`: both conflicting sections (plus any third section that states one of the two values). `missing_info`: the section(s) that state which information is required. `no_evidence`: empty. Never an expired document. |
| `key_facts` | `normal` only: short facts the reply must contain, written as in the KB. Every number in a key fact must appear in a gold section, except results of a calculation (tag `arithmetic`), which are explained in `reference`. Empty for the other types. |
| `missing_slots` | `missing_info` only: the items to ask for, at most three, in Vietnamese. |
| `reference` | `normal`: a model reply in Vietnamese following `QT-MAU-PHAN-HOI-2026` (≤ 120 words). Other types: the expected behaviour (what to ask, why to escalate, which values conflict). |
| `tags` | From the fixed vocabulary in section 5. |
| `conflict_id` | `conflict` only: the registry id (C01-C06). |
| `split` | `dev` or `test`, assigned by `scripts/split_tickets.py`, never by hand. |

## 4. Near-misses (must not be labelled conflict)

- The expired policy differs from the active one (`expired_trap`): answer from `CS-DOI-TRA-2025`.
- Same topic, different attribute: manufacturer warranty period vs store repair time; extended-warranty start vs manufacturer period.
- Narrower scope outside its range: a phone or laptop bought in October is not part of the October promotion, so the 7-day / 10% rule applies; an order of VND 600,000 at 15 km is free under the policy and outside the notice.
- A section of a variant document that does not carry the planted value (for example delivery time slots in `TB-GIAO-HANG-2026`) is ordinary evidence.
- Channel-specific offers (Daikin E-Shop Premier Care) and promotions that have ended (Samsung 3-year TV warranty, 01/04-30/06/2026).

## 5. Tag vocabulary

| Tag | Meaning |
|---|---|
| `no_diacritics` | Ticket typed without Vietnamese diacritics |
| `multi_doc` | Gold evidence spans two or more documents |
| `catalog` | Evidence from a product catalogue (`SP-`) |
| `manufacturer` | Evidence from a manufacturer warranty document (`NSX-`) |
| `simulated` | Evidence from a simulated operating document |
| `arithmetic` | The answer needs a calculation |
| `date_reasoning` | The answer needs date or working-day reasoning |
| `near_miss` | Looks like a conflict or a gap but is answerable (section 4) |
| `expired_trap` | The expired policy would give a different answer |
| `safety` | Safety signs that must be handled first |
| `email` | Long, email-style message with context or noise |

## 6. Quality rules

- Each ticket is written and labelled once; `scripts/validate_tickets.py` must pass before the ticket is kept.
- **LLM cross-check:** `scripts/llm_label_check.py` asks a stronger model than the system's generator (`JUDGE_MODEL`) to label every ticket blind: it sees the ticket, the decision procedure of section 2 and the KB (table of contents of all valid documents plus the full text of the retrieved candidate sections), never the reference label. Agreement and Cohen's kappa are reported, and every disagreement is adjudicated against this guide: the label is kept, the label is corrected, or the ticket is reworded when it is ambiguous.
- Labels are frozen together with the KB snapshot. Changing a KB value later requires re-running the validator and re-checking every ticket that cites the changed section.
