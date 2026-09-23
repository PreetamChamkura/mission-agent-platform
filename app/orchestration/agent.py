"""Agent definitions.

Each agent runs a small, auditable loop: retrieve -> guardrail-check ->
reason -> guardrail-check output -> record every step to a DecisionTrace.
Agents are role-scoped (their `role` determines what the guardrail policy
lets them do) and clearance-scoped (what classification of data they may
even retrieve).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.anomaly.detector import DETECTOR
from app.audit.trace import DecisionTrace, TRACE_STORE
from app.guardrails.policy import GuardrailEngine, GuardrailViolation, Verdict
from app.orchestration.llm import LLMClient, MockLLM
from app.rag.index import Indexer

CLEARANCE_LEVELS = {"PUBLIC": 0, "CUI": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP SECRET": 4}


@dataclass
class AgentResult:
    task_id: str
    status: str
    answer: str | None
    trace: dict


class BaseAgent:
    role: str = "research_agent"
    clearance: str = "CUI"

    def __init__(self, indexer: Indexer, guardrails: GuardrailEngine, llm: LLMClient | None = None):
        self.indexer = indexer
        self.guardrails = guardrails
        self.llm = llm or MockLLM()

    def run(self, task_id: str, query: str) -> AgentResult:
        trace: DecisionTrace = TRACE_STORE.create(task_id, self.role, query)
        try:
            clean_query, in_events = self.guardrails.check_input(query, agent_clearance=self.clearance)
            for e in in_events:
                trace.add("guardrail", f"input/{e.layer}: {e.verdict.value}", **{"reason": e.reason, **e.detail})

            self.guardrails.check_action(self.role, "search_docs", task_id)
            trace.add("tool_call", "search_docs", query=clean_query)

            max_level = CLEARANCE_LEVELS.get(self.clearance, 1)
            hits = self.indexer.search(clean_query, top_k=5, max_classification_level=max_level)
            for h in hits:
                trace.add(
                    "evidence", f"retrieved {h.doc.doc_id} (score={h.score:.3f})",
                    source_system=h.doc.source_system, record_type=h.doc.record_type,
                    classification=h.doc.classification, title=h.doc.title, score=round(h.score, 4),
                )
                DETECTOR.ingest(f"retrieval_score::{self.role}", h.score)

            evidence_block = "\n".join(f"- [{h.doc.source_system}] {h.doc.title}: {h.doc.text[:180]}" for h in hits)
            trace.add("thought", f"synthesizing answer from {len(hits)} evidence item(s)")
            answer_text = self.llm.complete(system=f"You are a {self.role}.", prompt=evidence_block)

            payload = {"answer": answer_text, "evidence_count": len(hits), "sources": [h.doc.doc_id for h in hits]}
            out_events = self.guardrails.check_output(payload, required_fields=["answer", "evidence_count"])
            for e in out_events:
                trace.add("guardrail", f"output/{e.layer}: {e.verdict.value}", reason=e.reason)

            confidence = min(0.95, 0.3 + 0.13 * len(hits))
            trace.add("answer", answer_text, confidence=confidence, sources=payload["sources"])
            trace.finish("completed", confidence=confidence)
            return AgentResult(task_id=task_id, status="completed", answer=answer_text, trace=trace.to_dict())

        except GuardrailViolation as gv:
            trace.add("guardrail", f"BLOCKED at {gv.event.layer}: {gv.event.reason}", verdict=gv.event.verdict.value)
            status = "requires_approval" if gv.event.verdict == Verdict.REQUIRE_APPROVAL else "denied"
            trace.finish(status)
            return AgentResult(task_id=task_id, status=status, answer=None, trace=trace.to_dict())


class ResearchAgent(BaseAgent):
    role = "research_agent"
    clearance = "CONFIDENTIAL"


class ComplianceAgent(BaseAgent):
    role = "compliance_agent"
    clearance = "SECRET"


class CoordinatorAgent(BaseAgent):
    role = "coordinator_agent"
    clearance = "CUI"


AGENT_REGISTRY = {
    "research_agent": ResearchAgent,
    "compliance_agent": ComplianceAgent,
    "coordinator_agent": CoordinatorAgent,
}
