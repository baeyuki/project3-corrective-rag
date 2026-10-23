# Week 3 - Knowledge base II, parsing and chunking

Sources: [week3_sources.md](week3_sources.md). Conflict registry: `data/eval/kb_conflicts.json`.

## 1. Weekly goals

- Extend the Week-2 knowledge base with product catalogues, one expired policy and controlled conflicting documents (~50 documents).
- Parse the corpus and split it into chunks that keep a stable citation `doc_id#section-id`.
- Report corpus statistics and check that the new documents are usable for evaluation.

## 2. Completed work

- **Product catalogues (10 documents, 35 models, 7 brands)** for phones, laptops, TVs, fridges, washers, air conditioners and kitchen appliances. Every value comes from an official manufacturer page (Apple, Samsung, LG, Lenovo PSREF, Daikin, Panasonic, Philips) read on 2026-10-22; values a page does not publish are marked as missing, never estimated.
- **Manufacturer warranty documents (6)**: periods, start-date rules, exclusions per brand.
- **One expired policy** (`CS-DOI-TRA-2024`), same section ids as the active policy but different values, replaced by `CS-DOI-TRA-2025`.
- **Six planted conflicts**, each issued by a different department (promotion terms, operations, service centre, instalment and accounting FAQs, technical notice) and contradicting an active policy on one value. All are registered in `data/eval/kb_conflicts.json` with their scope, together with compatible near-misses.
- **Two governance procedures** that make the expected behaviour part of the corpus: do not use expired documents, escalate when two active documents disagree, do not guess when information is missing; identify the exact model before answering.
- **13 simulated operating documents** (extended warranty, membership, pre-orders, delivery fees, installation prices and scheduling, safety incidents, business orders, reply standard, store network). Values are calibrated on public price references and must not contradict existing documents; all are labelled `provenance: simulated`.
- **Parser and chunker** (`corpus.py`, `chunking.py`): chunks never cross a section, long sections are split at 120 words on paragraph or bullet boundaries, each chunk starts with "document title > section heading" and carries status and validity dates.
- **Scripts and tests**: `build_corpus.py` (statistics), `probe_conflicts.py` (are conflicts retrievable?), 14 new tests.

## 3. Results / metrics

| Category | Documents | Sections | Chunks (120 words) |
|---|---:|---:|---:|
| Policy, procedure, FAQ, troubleshooting | 40 | 165 | 165 |
| Product catalogue | 10 | 58 | 86 |
| Manufacturer warranty | 6 | 24 | 25 |
| Promotion, notice | 4 | 16 | 16 |
| Price list, store directory | 3 | 12 | 12 |
| **Total** | **63** | **275** | **304** |

- 17,400 words; 62 documents valid on the snapshot date, 1 expired.
- Chunk granularities for Week 5: one per section (275), 120 words (304), 60 words (364).
- Both sides of every planted conflict rank in the BM25 top 10 for an accent-free query; the expired policy ranks 2nd for change-of-mind questions, so a validity filter is required.
- 25 tests pass, including a check that every number in the conflict registry appears in the cited document.

## 4. Issues & resolutions
| Issue | Resolution |
|---|---|
| The plan estimated 1,000-1,500 chunks; with real data the corpus gives 304 (reaching 1,000 would need chunks of about 12-17 words) | Chunk size is driven by retrieval quality, not by count; measured values are reported and three granularities are kept for comparison in Week 5 |
| Some official pages omit values or differ by region (e.g. iPhone SIM type, MacBook power adapter) | Missing values are written as missing; the region of each source page is stated in the catalogue |
| Lenovo per-model pages could not be read (rendered by JavaScript) | Used the official PSREF platform specification PDFs; the catalogue asks for the machine-type model before answering configuration questions |
| Official sources contain their own inconsistencies (Samsung warranty start date, LG unit codes) | Kept one rule and documented the discrepancy in the source registry |
| Risk of unintended conflicts when adding documents (e.g. delivery notice vs fee table, e-warranty activation vs change-of-mind returns) | Documents refer to existing policies instead of repeating values; remaining overlaps are registered; tests check the registry against the text |


## 5. Plan for next week (Week 4)

- Build the labelled evaluation set (~150 tickets) covering normal, missing_info, no_evidence and conflict cases, using the planted conflicts, the expired policy and the near-misses.
- Write a label guide and automatic label checks (gold sections exist and are valid on the snapshot date, key facts appear in the evidence).
- Create a reproducible dev/test split, with conflicts held out from tuning.
