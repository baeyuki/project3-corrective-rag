# Week 4 - Evaluation set: labelled tickets, label checks and dev split

Label guide: [week4_label_guide.md](week4_label_guide.md). Data: `data/eval/tickets.jsonl`.

## 1. Weekly goals

- Build a labelled set of ~150 Vietnamese support tickets covering the four outcomes: normal (answer), missing_info (clarify), no_evidence (escalate), conflict (escalate).
- Define labels precisely enough to be reproduced from the ticket, the guide and the KB alone, and check them automatically and with an independent LLM.
- Create a dev split for tuning and a test split that is never used for tuning.

## 2. Completed work

- **Label guide**: a 4-question decision procedure (conflict? no evidence? missing information? otherwise normal), fixed ticket date 2026-10-22, rules for every field, near-miss cases and a tag vocabulary.
- **168 tickets** written in Vietnamese chat and email style (abbreviations, no diacritics, long emails). Each ticket has gold sections (`doc_id#section-id`), key facts, a reference reply or expected behaviour, and difficulty tags. Real customer messages were not used because they contain personal data.
- **Extended ticket schema** with `tags`, `conflict_id` and `split`; missing_info tickets now cite the section that states the required information.
- **Automatic label validator** (`tickets.py`, `validate_tickets.py`) with 9 rules, e.g. gold sections must exist and be valid on the ticket date, every number in a key fact must appear in the gold text, conflict tickets must cite both sides of a registered conflict.
- **Grouped, stratified, reproducible dev/test split** (`split_tickets.py`).
- **LLM cross-check** (`llm_label_check.py`): a stronger model than the system's generator (`JUDGE_MODEL`, gpt-4.1-mini) labels all 168 tickets blind, seeing only the ticket, the decision procedure, a table of contents of the KB and the retrieved candidate sections; agreement, Cohen's kappa and the list of disagreements are reported.
- 8 new tests (33 in total), including a test that the cross-check prompt never contains the reference label.

## 3. Results / metrics

| Type | Decision | Tickets | dev | test |
|---|---|---:|---:|---:|
| normal | answer | 94 | 28 | 66 |
| missing_info | clarify | 26 | 9 | 17 |
| no_evidence | escalate | 23 | 7 | 16 |
| conflict | escalate | 25 | 8 | 17 |
| **Total** | | **168** | **52** | **116** |

- Hard cases tagged: 47 multi-document, 29 without diacritics (17%), 17 near-misses, 12 date reasoning, 10 arithmetic, 4 expired-policy traps, 4 safety.
- All six planted conflicts are covered (3-5 tickets each); C04 and C05 are in dev, C01, C02, C03 and C06 appear only in test.
- All 9 label rules pass for 168 tickets; the split is reproducible (`split_tickets.py --check`).
- First retrieval check with BM25 (gold section in top 5): normal 93%, missing_info 58%, both sides of a conflict 64%, so the set is not trivially easy.
- **LLM cross-check (gpt-4.1-mini, blind, 168 tickets):** type agreement 106/168 (0.63), Cohen's kappa 0.27. Agreement is high on normal tickets (88/92) and low on the other three types: the model labelled 23/25 conflict, 16/23 no_evidence and 19/28 missing_info tickets as normal. When both labelled a ticket normal, the model cited a gold section in 88/88 cases.

## 4. Issues & resolutions

| Issue | Resolution |
|---|---|
| No real customer tickets can be used (personal data, fictional retailer) | Tickets written by the project in realistic chat/email style; documented as a limitation |
| Some normal tickets legitimately cite both sides of a conflict (the case lies outside the conflict range) and were flagged as errors | Added the rule "allowed only with the `near_miss` tag" |
| "Bảo hành bao lâu" is ambiguous: warranty period or repair time (the latter is conflict C04) | Reworded three tickets to "được bảo hành mấy tháng" |
| Tickets written by the KB authors reuse its vocabulary, which may make retrieval look easier than in reality | More unaccented tickets added (29); paraphrased variants of the tickets can be generated later if retrieval looks too easy |
| Labels were written by a single author | Independent LLM cross-check on all 168 tickets (results in section 3); each disagreement adjudicated against the guide, 2 labels changed |
| The model resolves conflicts instead of flagging them: in 23 conflict tickets it chose one side, treating the newer notice or the promotion as overriding the policy, although it cited both conflicting sections in 19 of them | Labels kept: the KB defines no precedence (`QT-SU-DUNG-TAI-LIEU-2026`). Added the explicit rule "no precedence between documents" to the guide. This is the behaviour Corrective RAG must correct (also seen in the Week-1 proof of concept) |
| The model answers "the document does not say" as if it were an answer: in 10 of its 16 normal labels on no_evidence tickets, its own reason says the information is not in the documents; in 2 cases (Daikin unit weight, microwave timer) it claimed the value was present | Labels kept; added the rule "stated absence is not evidence" to the guide |
| The model answers generally instead of asking: 19 missing_info tickets labelled normal, 5 of them with a reason that itself names the missing information | Kept where the KB says to ask back or where the customer asks for an action on their own case; changed 2 tickets where the KB answers every variant; added the "variants" rule to the guide |
| The cross-check runs on all tickets, including test | Used only to audit labels, not to tune any system component; labels were changed only when the KB contradicted them, never to match the model |

## 5. Plan for next week (Week 5)

- Implement BM25, embedding (OpenAI `text-embedding-3-small`) and hybrid RRF retrieval with a validity filter for expired documents.
- Evaluate on the ticket set at section level with Recall@K, MRR and nDCG, plus "both sides retrieved" for conflict tickets; compare the three chunk granularities from Week 3.
- Choose the configuration on dev and record test results as baselines.
