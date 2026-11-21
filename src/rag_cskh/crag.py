"""Corrective RAG, part I (Week 7): analyse -> multi-query retrieve -> rerank -> grade (+ pairwise conflict check).

``CorrectiveRetriever.run`` returns the analysis, the candidate lists of every stage and the grade with its
decision. Week 8 adds query rewriting when the grade finds the evidence insufficient, the decision branches
(answer with the Week-6 shared prompt, ask, escalate) and the citation check.
"""

from __future__ import annotations

from rag_cskh.analysis import analyze
from rag_cskh.evidence import SectionStore
from rag_cskh.grader import K_GRADE, grade
from rag_cskh.multiquery import multi_query_sections
from rag_cskh.rerank import RERANK_DEPTH, LLMReranker


class CorrectiveRetriever:
    def __init__(self, retriever, store: SectionStore, llm, reranker: LLMReranker | None = None, grader_llm=None,
                 depth: int = RERANK_DEPTH, k_grade: int = K_GRADE, multi_query: bool = True, pairwise: bool = True):
        self.retriever, self.store, self.llm = retriever, store, llm
        self.reranker = reranker or LLMReranker(llm, store)
        self.grader_llm = grader_llm or llm
        self.depth, self.k_grade, self.multi_query, self.pairwise = depth, k_grade, multi_query, pairwise

    def candidates(self, ticket: str, analysis) -> list[str]:
        if self.multi_query:
            return multi_query_sections(self.retriever, analysis.queries, self.depth)
        return self.retriever.search_sections(ticket, self.depth)

    def run(self, ticket: str) -> dict:
        analysis = analyze(self.llm, ticket)
        fused = self.candidates(ticket, analysis)
        ranked, grades = self.reranker.rerank(ticket, fused)
        graded = ranked[:self.k_grade]
        g = grade(self.grader_llm, analysis, graded, self.store, self.pairwise)
        return {"analysis": analysis.to_dict(), "queries": analysis.queries, "fused": fused, "ranked": ranked,
                "rerank_grades": grades, "graded": graded, "grade": g.to_dict(), "decision": g.decision}
