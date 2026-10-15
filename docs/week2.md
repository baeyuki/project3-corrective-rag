# Week 2 - Knowledge base I

**Goal:** build the first usable, source-grounded Vietnamese knowledge base for the fictional electronics retailer, covering policies, operating procedures, frequently asked questions and troubleshooting guides.

**Corpus snapshot:** 2026-10-15. The source registry and evidence limitations are documented in [week2_sources.md](week2_sources.md).

## 1. Deliverable

The corpus contains 25 Markdown documents in `data/kb/`. Files beginning with `_`, such as `_TEMPLATE.md`, are authoring aids and are not part of the corpus.

| Category | Prefix | Documents | Purpose |
|---|---|---:|---|
| Policy | `CS-` | 7 | Rules, eligibility, limits, fees and response times |
| Procedure | `QT-` | 6 | Steps and required information for support agents |
| FAQ | `FAQ-` | 5 | Direct answers to common customer questions |
| Troubleshooting | `HD-` | 7 | Safe diagnostic steps for common product problems |
| **Total** | | **25** | |

All 25 documents are active and use an effective date of 2026-01-01. Expired documents and deliberately conflicting rules are reserved for Week 3 so this release provides a clean baseline.

## 2. Topic coverage

### Policies

- Warranty eligibility, exclusions and handling time.
- Returns, exchanges and refunds.
- Delivery, inspection and basic installation.
- Payment methods and payment safety.
- Instalments through credit cards and finance companies.
- Customer-data privacy and identity verification.
- Complaint intake, response targets and escalation.

### Procedures

- Looking up and confirming an order.
- Cancelling an order at different fulfilment stages.
- Receiving a product for warranty service.
- Assessing and recording a return request.
- Handling late, damaged, missing or incorrectly delivered orders.
- Issuing and correcting electronic invoices.

### FAQs

The FAQ set covers orders, warranty, returns, delivery and payment. It gives short customer-facing answers while preserving the information requirements needed for `missing_info` decisions, such as order ID, phone number, product model and delivery address.

### Troubleshooting

The guides cover washing machines, refrigerators, TVs, air conditioners, laptops, phones and small home appliances. Each guide contains safe checks that customers can perform and explicit stop/escalation conditions for electricity, heat, smoke, water ingress and battery swelling.

## 3. Authoring conventions

Every document follows `data/kb/_TEMPLATE.md`:

- UTF-8 Vietnamese content.
- YAML-like front matter with `doc_id`, `title`, `category`, `status` and `effective_date`.
- A unique `doc_id` equal to the filename.
- Stable lowercase ASCII section anchors in the form `{#section-id}`.
- One support topic per `##` section.
- Concrete thresholds and time windows where a rule depends on them.
- A provenance class and one or more references into `week2_sources.md`.
- No source citations inside the fictional corpus; future evaluation labels cite sections as `doc_id#section-id`.

The corpus currently contains 99 independently addressable sections. Most documents have four sections; the order-confirmation procedure has three.

## 4. Design decisions

- **Information requirements are stated explicitly.** Procedures say which fields an agent must request, allowing a grader to distinguish `missing_info` from `no_evidence`.
- **Policies and procedures have different roles.** Policies define what is allowed; procedures define how an agent verifies and performs it.
- **Safety boundaries are explicit.** Troubleshooting never asks customers to open high-voltage equipment and identifies conditions that require stopping use.
- **Rules are consistent in this release.** Repeated values such as the 7-day change-of-mind window and the 5-to-10-day refund period agree across policy and FAQ documents. Controlled conflicts will be planted and labelled in Week 3.
- **Section anchors are semantic and immutable.** Later parsing can retain them as citation IDs even if headings or prose are edited.

## 5. Research and provenance method

The corpus is not presented as the policy set of a real retailer. It is a controlled synthetic benchmark derived from official public material:

1. Official Vietnamese retailer pages establish realistic policy dimensions: eligibility, category exceptions, condition checks, delivery zones, payment responsibilities and instalment-provider dependence.
2. Official manufacturer and platform support pages establish safe troubleshooting sequences and stop conditions.
3. Vietnamese government and regulator sources establish the privacy, consumer-protection, invoicing and payment-safety boundaries.
4. Rules are normalised into one internally consistent fictional company so evaluation has a single answer except where Week 3 deliberately plants a conflict.
5. Internal agent procedures are labelled `operational-synthesis`: their public-policy basis is sourced, while their step order and service targets are disclosed research assumptions.

Every KB document includes `provenance` and `source_refs` metadata. The registry records authority, use and adaptation for each source. This permits a reviewer to trace the rationale without confusing synthetic company rules with verbatim policies of the referenced organisations.

## 6. Validation

`tests/test_kb.py` checks:

- exactly 25 indexable documents and the intended category distribution;
- required metadata, valid dates, active status and unique document IDs;
- filename-to-`doc_id` consistency;
- at least three non-empty sections per document;
- globally unique `doc_id#section-id` references;
- valid UTF-8 text without replacement characters and basic Vietnamese topic coverage.
- valid provenance values and source references that resolve to the source registry.

Run all checks with:

```bash
pytest -q
```

## 7. Handoff to Week 3

Week 3 can extend this baseline with product catalogues, one clearly expired policy and controlled conflicting documents. The parser should treat each `##` section as the minimum citation unit, preserve front matter and section anchors as metadata, and then evaluate whether additional token-based splitting is necessary to reach the planned chunk volume.
