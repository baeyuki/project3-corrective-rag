# Week 1 - Survey of similar applications and related research

**Goal:** review existing AI customer-support products and the research behind the AI components of this project, then fix the project's scope.

## 1. Similar applications

| Product | Who it helps | Knowledge grounding | Safety / handoff |
|---|---|---|---|
| **Intercom Fin AI Agent** | Customers (end-to-end resolution on chat, email, voice) | Help-center articles, internal content, PDFs, web pages; live data from Shopify, Stripe, Salesforce | Reranks content to down-rank outdated sources; validation step on answers; configurable escalation to humans |
| **Zendesk Copilot** | Human agents | Drafts replies from macros and help-center articles | Agent edits and sends; guided procedures for multi-step cases |
| **Salesforce Service Replies (Agentforce for Service)** | Human agents | Grounded on knowledge articles and related records | Agent can post, edit or reject each suggestion |
| **Google Cloud Agent Assist** | Human agents in contact centres | (Generative) knowledge assist over company documents | Suggestions only; also smart reply, summarisation |
| **FPT AI Agents (Vietnam)** | Customers on web, app, Facebook, Zalo | Uploaded documents (mainly PDF) split into editable chunks | Not documented on the overview page |

**Observations**
- Two modes exist: customer-facing agents that answer directly, and agent-assist copilots that draft replies for a human. Both ground answers in a company knowledge base, i.e. they are RAG systems.
- Leading products add checks beyond plain RAG: down-ranking outdated content, validating answers, explicit escalation rules.
- What is not public: how these checks work and how well they perform. None publish a benchmark, and explicit handling of *contradicting* documents is not documented.
- Vietnamese support exists (FPT AI Agents), but with no public description of the retrieval quality for Vietnamese text.

## 2. Related research

| Work | Key idea | Use in this project |
|---|---|---|
| Lewis et al. 2020 - RAG | Combine a retriever with a generator to answer from external knowledge | Common basis of all compared systems |
| Yan et al. 2024 - Corrective RAG | A retrieval evaluator labels results correct / incorrect / ambiguous and triggers corrective actions | Evidence-grading step and query-rewriting loop |
| Asai et al. 2024 - Self-RAG | The model decides when to retrieve and critiques retrieved passages and its own output | LLM self-assessment of relevance and support |
| Robertson & Zaragoza 2009 - BM25 | Probabilistic keyword ranking | Keyword retrieval (error codes, model names, numbers) |
| Cormack et al. 2009 - Reciprocal Rank Fusion | Fuse rankings with sum of 1/(k + rank) | Combine BM25 and vector results, and several queries |
| Chen et al. 2024 - BGE M3 | Multilingual embeddings and rerankers | Cross-encoder reranker for Vietnamese |
| Zheng et al. 2023 - LLM-as-a-Judge | Strong LLMs agree well with human raters but show self-preference bias | Judge model stronger than the generator |
| Es et al. 2024 - RAGAS | Reference-free RAG metrics (faithfulness, context precision, ...) | Metric definitions |

## 3. Takeaways and project scope

- **Mode:** agent-assist. The system drafts a cited reply and handling steps; a human agent sends it.
- **Language:** the knowledge base, the tickets and every user-facing output (reply to the customer, handling steps, clarifying questions, escalation reason) are in Vietnamese. Code and project documentation are in English.
- **Beyond CRAG:** the original CRAG fixes poor retrieval. Support tickets also need two more outcomes, *ask the customer* (missing information) and *escalate* (no evidence or conflicting documents), so grading has five labels: sufficient, insufficient, missing_info, no_evidence, conflict.
- **Vietnamese retrieval:** words span several syllables and customers often type without diacritics, so BM25 will use syllable unigrams, bigrams and accent-stripped tokens, combined with multilingual embeddings.
- **Evaluation gap:** commercial tools publish no benchmark, so the project builds a labelled ticket set covering all four outcomes and compares Corrective RAG with Vector RAG and Hybrid RAG on retrieval, answer, safety and cost metrics.

## 4. Feasibility checks

Two small proofs of concept on a toy corpus of 8 passages (`data/poc/`).

**Vietnamese tokenisation for BM25** (`python scripts/poc_tokenization.py`, 12 queries, half typed without diacritics):

| Tokeniser | Hit@1 / MRR, with diacritics | Hit@1 / MRR, without diacritics |
|---|---|---|
| Whitespace words | 0.83 / 0.89 | 0.50 / 0.50 |
| Syllables + bigrams | 0.83 / 0.89 | 0.50 / 0.50 |
| + accent-folded copies (both sides) | 0.83 / 0.89 | **0.83 / 0.89** |

- Folding diacritics on both the document and the query side recovers queries typed without accents.
- The remaining miss ("bảo hành mất bao lâu") ranks the borrowing rule and the *expired* 2023 policy above the right passage: keyword retrieval alone is not enough, which motivates the reranker and the evidence-grading step.

**LLM evidence grading** (`python scripts/poc_grading.py`): four toy tickets, one per expected verdict, top-4 BM25 passages, graded by `gpt-4o-mini` with Structured Outputs. Result: **2/4 correct**, 1,712 tokens in total.

| Ticket | Expected | Got | Note |
|---|---|---|---|
| Warranty over 20 working days, can I borrow a device? | sufficient | sufficient | Cited both warranty passages |
| Cancel my order | missing_info | no_evidence | The toy corpus has no order-handling rule, so the model could not know an order number is required |
| Can I rent a projector? | no_evidence | no_evidence | |
| Return period for October promotion purchases | conflict | sufficient | Both conflicting passages (7 vs 15 days) were retrieved; the model picked the promotion rule and ignored the general policy |

Lessons for the design:
- **Conflict detection is the hard part.** Retrieval was not the problem; the grader needs an explicit step that lists the rule each passage states and compares them before deciding.
- **missing_info depends on the knowledge base:** the model can only ask for an order number if a document says one is needed. The real knowledge base must include order-handling procedures, and the label guide must state this.
- Structured Outputs returned valid JSON for every call, and the reasons were in Vietnamese as required.

## 5. Week-1 deliverables

- This survey.
- Draft data formats: `data/kb/_TEMPLATE.md` (knowledge-base document) and `data/eval/ticket_schema.json` with one example ticket per outcome.
- Vietnamese tokeniser and BM25 (`src/rag_cskh/`), the two proofs of concept above, `scripts/check_env.py`.
- 12-week plan: [plan.md](plan.md).
- `pytest`: 6 tests (data formats, tokeniser, BM25).

## Sources

- Fin: [What is Fin AI Agent?](https://fin.ai/learn/what-is-fin-ai-agent)
- Zendesk Copilot: [overview](https://www.eesel.ai/blog/what-is-zendesk-copilot)
- Salesforce: [What are Service Replies](https://www.salesforce.com/blog/what-are-salesforce-service-replies/)
- Google Cloud: [Agent Assist concepts](https://docs.cloud.google.com/gemini-enterprise-cx/agent-assist/concepts)
- FPT: [FPT AI Agents documentation](https://docs.fpt.ai/docs/vi/aiagents/documentation/aiagent)
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Yan et al. (2024). Corrective Retrieval Augmented Generation. arXiv:2401.15884.
- Asai et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Cormack, Clarke & Büttcher (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
- Chen et al. (2024). BGE M3-Embedding. arXiv:2402.03216.
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS.
- Es et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation. EACL (demo).
