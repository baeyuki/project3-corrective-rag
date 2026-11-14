"""Week-6 tests: LLM cache, reranker ordering, shared answer prompt, baselines and answer metrics (offline)."""

import json

from rag_cskh import answer, rerank
from rag_cskh.answer_metrics import covered, macro_f1, ticket_answer_metrics
from rag_cskh.corpus import load_corpus
from rag_cskh.embeddings import HashingEmbedder
from rag_cskh.evidence import SectionStore
from rag_cskh.llm import FakeLLM
from rag_cskh.pipelines import build_baselines
from rag_cskh.tickets import load_tickets

DOCS = load_corpus()
STORE = SectionStore(DOCS)


def test_llm_cache_calls_once_and_keeps_usage(tmp_path):
    calls = []

    def respond(messages, schema):
        calls.append(1)
        return {"x": len(calls)}

    llm = FakeLLM(respond, tmp_path)
    msgs = [{"role": "user", "content": "xin chào"}]
    a, _ = llm.chat_json(msgs, {"type": "object"})
    b, _ = FakeLLM(respond, tmp_path).chat_json(msgs, {"type": "object"})   # new process, same cache file
    assert a == b == {"x": 1} and len(calls) == 1
    assert llm.usage.calls == 1 and llm.usage.prompt_tokens == 2


def test_rerank_orders_by_grade_and_keeps_every_candidate():
    cands = ["a", "b", "c", "d"]
    ranked, grade = rerank.order_by_scores(cands, [{"id": 3, "score": 3}, {"id": 1, "score": 1},
                                                   {"id": 2, "score": 3}, {"id": 9, "score": 3}])
    assert ranked == ["b", "c", "a", "d"]          # ties keep first-stage order; "d" was not graded
    assert grade["d"] == -1 and sorted(ranked) == sorted(cands)


def test_rerank_prompt_numbers_sections_and_hides_citations():
    cits = ["CS-DOI-TRA-2025#doi-do-nhu-cau", "KM-THANG-10-2026#doi-tra"]
    msgs = rerank.build_messages("đổi hàng khuyến mãi", cits, STORE)
    user = msgs[1]["content"]
    assert "[1] " in user and "[2] " in user and "#doi-tra" not in user


def test_answer_prompt_never_contains_labels():
    t = next(t for t in load_tickets() if t["type"] == "normal" and t["split"] == "dev")
    msgs = answer.build_messages(t["ticket"], t["gold_sections"], STORE)
    text = json.dumps(msgs, ensure_ascii=False)
    assert t["reference"] not in text and "expected_decision" not in text
    assert all(f"[{c}]" in text for c in t["gold_sections"])


def test_citations_are_normalised_and_checked_against_context():
    valid, invalid = answer.clean_citations(["[CS-BAO-HANH-2025#tu-choi]", "FAQ-X#y", "CS-BAO-HANH-2025#tu-choi"],
                                            ["CS-BAO-HANH-2025#tu-choi"])
    assert valid == ["CS-BAO-HANH-2025#tu-choi"] and invalid == ["FAQ-X#y"]


def test_lexical_coverage_handles_missing_diacritics():
    assert covered("không bảo hành miễn phí", "Dạ, sản phẩm rơi vỡ khong duoc bao hanh mien phi ạ")
    assert not covered("số sê-ri", "Anh/chị cho em xin mã đơn")
    assert covered("kích thước chỗ đặt (rộng, cao, sâu)", "Anh/chị cho em kích thước chỗ đặt tủ nhé")


def test_answer_metrics_on_conflict_ticket():
    t = {"type": "conflict", "expected_decision": "escalate", "gold_sections": ["A#x", "B#y"], "key_facts": [],
         "missing_slots": []}
    out = {"decision": "escalate", "reply": "Dạ em đã chuyển yêu cầu CS-DOI-TRA-2025 ...", "citations": ["A#x", "B#y"],
           "missing_info": [], "invalid_citations": []}
    m = ticket_answer_metrics(t, ["A#x", "B#y", "C#z"], out)
    assert m["decision_ok"] == m["context_ok"] == m["cited_ok"] == 1.0 and m["code_leak"] == 1.0
    assert macro_f1(["answer", "clarify"], ["answer", "answer"]) == (2 / 3 + 0 + 0) / 3


def test_baselines_run_end_to_end_offline(tmp_path):
    seen = []

    def respond(messages, schema):
        seen.append(schema)
        if "scores" in schema["properties"]:
            n = messages[1]["content"].count("\n\n[") + 1
            return {"scores": [{"id": i, "score": 3 if i == n else 1} for i in range(1, n + 1)]}
        return {"decision": "answer", "reply": "Dạ em chào anh/chị.", "citations": [], "missing_info": []}

    llm = FakeLLM(respond, tmp_path)
    systems = build_baselines(DOCS, HashingEmbedder(), llm)
    out_v = systems["vector-rag"].run("iphone air co khe sim vat ly khong")
    out_h = systems["hybrid-rag"].run("iphone air co khe sim vat ly khong")
    assert len(out_v["evidence"]) == len(out_h["evidence"]) == 5
    assert len(out_h["first_stage"]) == 20
    assert out_h["evidence"][0] == out_h["first_stage"][-1]          # the reranker's top grade moved to rank 1
    assert not any(c.startswith("CS-DOI-TRA-2024") for c in out_h["first_stage"])   # validity filter kept
    assert out_v["answer"]["decision"] == "answer"


def test_cross_encoder_reranker_orders_caches_and_segments(tmp_path):
    from rag_cskh.cross_encoder import CrossEncoderReranker, passage, resolve

    seen = []

    def predict(pairs):
        seen.extend(pairs)
        return [float("> iphone air" in p[1].lower()) for p in pairs]

    cits = ["SP-DT-APPLE-2026#iphone-16", "SP-DT-APPLE-2026#iphone-air", "SP-DT-APPLE-2026#iphone-17"]
    ce = CrossEncoderReranker(predict, STORE, "fake/ce", segment=str.upper, cache_dir=tmp_path)
    ranked, score = ce.rerank("iphone air co esim khong", cits)
    assert ranked[0] == "SP-DT-APPLE-2026#iphone-air" and ranked[1:] == [cits[0], cits[2]]   # ties keep order
    assert seen[0][0] == "IPHONE AIR CO ESIM KHONG" and seen[0][1] == passage(STORE, cits[0]).upper()
    again = CrossEncoderReranker(predict, STORE, "fake/ce", cache_dir=tmp_path).rerank("iphone air co esim khong", cits)
    assert len(seen) == 3 and again[1] == score                                               # served from cache
    assert resolve("bge-m3")["model"] == "BAAI/bge-reranker-v2-m3" and resolve("x/y")["model"] == "x/y"
    assert "hiệu lực" not in passage(STORE, cits[0])                                          # body only, no metadata line
