"""Week-7 tests: ticket analysis, multi-query fusion, evidence-grader decision rules and the CRAG stage (offline)."""

import json

from rag_cskh import analysis, grader
from rag_cskh.analysis import Analysis
from rag_cskh.chunking import chunk_corpus
from rag_cskh.corpus import load_corpus
from rag_cskh.crag import CorrectiveRetriever
from rag_cskh.embeddings import HashingEmbedder
from rag_cskh.evidence import SectionStore
from rag_cskh.llm import FakeLLM
from rag_cskh.multiquery import fuse_sections
from rag_cskh.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_cskh.tickets import SNAPSHOT, load_conflicts, load_tickets

DOCS = load_corpus()
STORE = SectionStore(DOCS)
REGISTRY = load_conflicts()
C01 = next(c for c in REGISTRY["conflicts"] if c["id"] == "C01")["sections"]
N02 = next(c for c in REGISTRY["compatible"] if c["id"] == "N02")["sections"]
OTHER = "FAQ-BAO-HANH-2025#mat-hoa-don"


def an(subs=("câu chính",)):
    return Analysis("ticket", "ticket", list(subs))


def out(sections=(), subs=(), missing=(), conflicts=()):
    return {"sections": [{"id": i, "relevance": r} for i, r in sections],
            "sub_questions": [{"index": i, "status": s, "evidence": e} for i, s, e in subs],
            "missing_info": [{"item": m, "required_by": r} for m, r in missing],
            "conflicts": [dict(zip(("a", "b", "attribute", "value_a", "value_b", "in_scope_of_both"), c)) for c in conflicts],
            "reason": "r"}


def test_analysis_queries_keep_raw_ticket_first_and_deduplicate():
    a = analysis.parse("doi tra hang km", {"normalized": "Đổi trả hàng khuyến mãi", "facts": [], "products": [],
                                           "sub_questions": ["Đổi trả hàng khuyến mãi", "Phí đổi?", "a", "b"]})
    assert a.queries[0] == "doi tra hang km" and a.queries.count("Đổi trả hàng khuyến mãi") == 1
    assert len(a.sub_questions) == 3
    empty = analysis.parse("Hủy đơn giúp em", {"normalized": "", "sub_questions": [], "facts": [], "products": []})
    assert empty.sub_questions == ["Hủy đơn giúp em"] and empty.queries == ["Hủy đơn giúp em"]


def test_fusion_rewards_sections_found_by_several_queries():
    fused = fuse_sections([["a", "b", "c"], ["c", "d"], ["c", "a"]])
    assert fused[0] == "c" and set(fused) == {"a", "b", "c", "d"}


def test_planted_conflict_escalates():
    g = grader.interpret(out([(1, "relevant"), (2, "relevant")], [(1, "answered", [1])],
                             conflicts=[(1, 2, "thời hạn đổi", "15 ngày", "7 ngày", True)]), an(), C01)
    assert g.decision == "escalate" and g.trigger == "conflict" and {g.conflicts[0]["a"], g.conflicts[0]["b"]} == set(C01)


def test_near_miss_and_invalid_conflicts_are_refuted_by_the_code():
    cits = [*N02, "CS-BAO-HANH-2025#tu-choi"]
    g = grader.interpret(out([(1, "relevant"), (2, "relevant"), (3, "irrelevant")], [(1, "answered", [1, 2])],
                             conflicts=[(1, 2, "thời hạn", "24 tháng", "15 ngày làm việc", False),   # out of scope
                                        (1, 1, "x", "a", "b", True),                                # same section
                                        (2, 3, "x", "a", "b", True),                                # same document
                                        (1, 3, "x", "a", "b", True),                                # side irrelevant
                                        (1, 9, "x", "a", "b", True)]), an(), cits)                  # bad number
    assert g.decision == "answer" and not g.conflicts
    assert sorted(r["rejected_because"] for r in g.rejected_conflicts) == sorted(
        ["customer case not in scope of both", "invalid section numbers", "same document", "a side graded irrelevant",
         "invalid section numbers"])


def test_decision_order_no_evidence_then_clarify_then_answer():
    cits = [OTHER, C01[0]]
    none = grader.interpret(out([(1, "irrelevant"), (2, "partial")], [(1, "not_in_documents", [])]), an(), cits)
    assert (none.decision, none.trigger) == ("escalate", "no_evidence")
    nothing = grader.interpret(out([(1, "irrelevant")], [(1, "answered", [1])]), an(), cits)
    assert nothing.decision == "escalate"                                  # nothing relevant at all
    ask = grader.interpret(out([(1, "relevant")], [(1, "needs_customer_info", [1])], [("ngày giao", 1)]), an(), cits)
    assert ask.decision == "clarify" and ask.missing_info[0]["required_by"] == OTHER
    no_item = grader.interpret(out([(1, "relevant")], [(1, "needs_customer_info", [1])]), an(), cits)
    assert no_item.decision == "answer"                                    # nothing concrete to ask -> answer
    side = grader.interpret(out([(1, "relevant")], [(1, "answered", [1]), (2, "not_in_documents", [])]),
                            an(("chính", "phụ")), cits)
    assert side.decision == "answer"                                       # only the main question decides escalation
    side_info = grader.interpret(out([(1, "relevant")], [(1, "answered", [1]), (2, "needs_customer_info", [1])],
                                     [("mã đơn", 1)]), an(("chính", "phụ")), cits)
    assert side_info.decision == "answer"                                  # ... and clarification (v2)
    missing_status = grader.interpret(out([(1, "relevant")], []), an(), cits)
    assert missing_status.decision == "escalate"                           # unanswered main question is not assumed answered


def test_conflict_beats_missing_info_and_evidence_order():
    cits = [C01[1], OTHER, C01[0]]
    g = grader.interpret(out([(1, "partial"), (2, "irrelevant"), (3, "relevant")], [(1, "needs_customer_info", [1])],
                             [("ngày giao", 1)], [(3, 1, "phí", "0%", "10%", True)]), an(), cits)
    assert g.decision == "escalate" and g.evidence == [C01[0], C01[1]]


def test_grader_prompt_numbers_sections_and_contains_no_labels():
    t = next(t for t in load_tickets() if t["type"] == "conflict" and t["split"] == "dev")
    msgs = grader.build_messages(Analysis(t["ticket"], t["ticket"], [t["ticket"]], ["giao 12/10"]), t["gold_sections"], STORE)
    text = json.dumps(msgs, ensure_ascii=False)
    assert "[1] " in text and "#" + t["gold_sections"][0].split("#")[1] + "]" not in text
    for leak in ("conflict_id", "expected_decision", t["reference"], "kb_conflicts"):
        assert leak not in text


def test_corrective_retriever_end_to_end_offline(tmp_path):
    chunks = chunk_corpus(DOCS, 120)
    hybrid = HybridRetriever([BM25Retriever(chunks, active_on=SNAPSHOT), DenseRetriever(chunks, HashingEmbedder(), SNAPSHOT)])

    def respond(messages, schema):
        props = schema["properties"]
        if "normalized" in props:
            return {"normalized": "Hàng khuyến mãi tháng 10 đổi được mấy ngày", "facts": [], "products": ["tivi"],
                    "sub_questions": ["Hàng khuyến mãi tháng 10 đổi do nhu cầu trong bao nhiêu ngày?",
                                      "Phí đổi hàng do nhu cầu là bao nhiêu?"]}
        n = messages[1]["content"].count("\n\n[") + 1
        if "pairs" in props:                         # pairwise check: the main call already found the conflict
            return {"pairs": []}
        if "scores" in props:
            return {"scores": [{"id": i, "score": 2} for i in range(1, n + 1)]}
        user = messages[1]["content"]
        nums = {c: i for i, c in enumerate(cands, 1)}
        ids = [nums[c] for c in C01 if c in nums]
        conflicts = [[ids[0], ids[1], "thời hạn đổi", "15 ngày", "7 ngày", True]] if len(ids) == 2 else []
        assert "CÂU HỎI CON" in user
        return out([(i, "relevant") for i in ids], [(1, "answered", ids)], conflicts=conflicts)

    cands: list[str] = []
    llm = FakeLLM(respond, tmp_path)
    crag = CorrectiveRetriever(hybrid, STORE, llm)
    orig = crag.reranker.rerank

    def capture(ticket, citations):
        ranked, g = orig(ticket, citations)
        cands[:] = ranked[:10]
        return ranked, g
    crag.reranker.rerank = capture
    res = crag.run("mua tivi dot khuyen mai thang 10 doi tra do khong thich duoc may ngay")
    assert res["queries"][0].startswith("mua tivi") and len(res["queries"]) == 4
    assert len(res["graded"]) == 10 and not any(c.startswith("CS-DOI-TRA-2024") for c in res["fused"])
    assert all(c in res["graded"] for c in C01)                      # multi-query keeps both sides of C01
    assert res["decision"] == "escalate" and res["grade"]["trigger"] == "conflict"


def test_pairwise_check_builds_cross_document_pairs_and_requires_every_judgment():
    cits = [C01[0], "KM-THANG-10-2026#san-pham-ap-dung", C01[1], OTHER]
    g = grader.interpret(out([(1, "relevant"), (2, "relevant"), (3, "partial"), (4, "irrelevant")], [(1, "answered", [1])]),
                         an(), cits)
    pairs = grader.candidate_pairs(g, cits)
    assert pairs == [(C01[0], C01[1]), ("KM-THANG-10-2026#san-pham-ap-dung", C01[1])]   # no same-doc, no irrelevant
    msgs = grader.build_pair_messages(an(), pairs, STORE)
    assert "Cặp 2: mục 3 (a) và mục 2 (b)" in msgs[1]["content"] and "#doi-tra]" not in msgs[1]["content"]
    judged = {"pairs": [
        {"pair": 1, "same_attribute": True, "attribute": "thời hạn đổi", "value_a": "15 ngày", "value_b": "7 ngày",
         "applies_a": True, "applies_b": True, "outcome_a": "đổi trong 15 ngày", "outcome_b": "đổi trong 7 ngày, phí 10%",
         "outcomes_differ": True},
        {"pair": 2, "same_attribute": True, "attribute": "x", "value_a": "a", "value_b": "b", "applies_a": False,
         "applies_b": True, "outcome_a": "a", "outcome_b": "b", "outcomes_differ": True},     # other channel: near-miss
        {"pair": 7, "same_attribute": True, "attribute": "x", "value_a": "a", "value_b": "b", "applies_a": True,
         "applies_b": True, "outcome_a": "a", "outcome_b": "b", "outcomes_differ": True}]}   # unknown pair number
    grader.apply_pair_checks(g, pairs, judged)
    assert g.decision == "escalate" and g.trigger == "conflict" and len(g.conflicts) == 1
    assert g.conflicts[0]["source"] == "pair_check" and [p["conflict"] for p in g.pair_checks] == [True, False]
    assert g.conflicts[0]["value_b"] == "đổi trong 7 ngày, phí 10%"


def test_pairwise_check_without_conflict_keeps_the_decision():
    cits = list(N02)
    g = grader.interpret(out([(1, "relevant"), (2, "relevant")], [(1, "answered", [1, 2])]), an(), cits)
    grader.apply_pair_checks(g, grader.candidate_pairs(g, cits), {"pairs": [
        {"pair": 1, "same_attribute": False, "attribute": "", "value_a": "", "value_b": "", "applies_a": True,
         "applies_b": True, "outcome_a": "", "outcome_b": "", "outcomes_differ": False}]})
    assert g.decision == "answer" and not g.conflicts and len(g.pair_checks) == 1


def test_same_threshold_outcome_is_not_a_conflict():
    # N01 near-miss: thresholds differ (300,000 vs 500,000 d) but both sections make a 600,000 d order free
    cits = ["TB-GIAO-HANG-2026#mien-phi-giao-hang", "CS-GIAO-HANG-2025#phi-giao-hang"]
    g = grader.interpret(out([(1, "relevant"), (2, "relevant")], [(1, "answered", [1, 2])]), an(), cits)
    grader.apply_pair_checks(g, grader.candidate_pairs(g, cits), {"pairs": [
        {"pair": 1, "same_attribute": True, "attribute": "ngưỡng miễn phí giao", "value_a": "300.000 đ, 10 km",
         "value_b": "500.000 đ, 20 km", "applies_a": True, "applies_b": True, "outcome_a": "miễn phí",
         "outcome_b": "miễn phí", "outcomes_differ": False}]})
    assert g.decision == "answer" and not g.conflicts


def test_identical_or_empty_outcomes_are_not_conflicts():
    cits = ["CS-THANH-TOAN-2025#chuyen-khoan", "FAQ-THANH-TOAN-2025#tru-tien", "FAQ-THANH-TOAN-2025#chua-xac-nhan"]
    g = grader.interpret(out([(1, "relevant"), (2, "relevant")], [(1, "answered", [1, 2])]), an(), cits)
    base = {"same_attribute": True, "attribute": "x", "applies_a": True, "applies_b": True, "outcomes_differ": True}
    grader.apply_pair_checks(g, grader.candidate_pairs(g, cits), {"pairs": [
        {**base, "pair": 1, "value_a": "a", "value_b": "b", "outcome_a": "Không thanh toán lại ngay",
         "outcome_b": "không  thanh toán lại ngay"}]})                      # same outcome, different case/spaces
    assert g.decision == "answer" and not g.conflicts
    g2 = grader.interpret(out([(1, "relevant"), (2, "relevant")], [(1, "answered", [1, 2])]), an(), cits)
    grader.apply_pair_checks(g2, grader.candidate_pairs(g2, cits), {"pairs": [
        {**base, "pair": 1, "value_a": "", "value_b": "b", "outcome_a": "x", "outcome_b": "y"}]})   # no value on side a
    assert not g2.conflicts
