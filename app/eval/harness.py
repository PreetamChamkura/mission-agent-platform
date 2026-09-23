"""Eval harness: runs the eval suite against the live agents and produces a
reliability report scored against mission requirements, not just generic
model metrics. Meant to run in CI before a new guardrail policy, prompt, or
model swap ships to a deployed environment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.eval.cases import EVAL_SUITE, EvalCase
from app.guardrails.policy import GuardrailEngine
from app.orchestration.agent import AGENT_REGISTRY
from app.rag.index import Indexer


@dataclass
class EvalOutcome:
    case_id: str
    requirement: str
    passed: bool
    detail: str
    latency_ms: float


@dataclass
class EvalReport:
    outcomes: list[EvalOutcome] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    @property
    def pass_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.passed) / len(self.outcomes)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "pass_rate": round(self.pass_rate, 3),
            "total": len(self.outcomes),
            "passed": sum(1 for o in self.outcomes if o.passed),
            "failed": sum(1 for o in self.outcomes if not o.passed),
            "outcomes": [o.__dict__ for o in self.outcomes],
        }


def run_eval_suite(indexer: Indexer, suite: list[EvalCase] = EVAL_SUITE) -> EvalReport:
    report = EvalReport()
    for case in suite:
        # fresh guardrail engine per case so rate limits from other cases don't bleed in
        guardrails = GuardrailEngine()
        agent_cls = AGENT_REGISTRY[case.agent_role]
        agent = agent_cls(indexer, guardrails)

        t0 = time.time()
        result = agent.run(task_id=f"eval::{case.case_id}", query=case.query)
        latency_ms = (time.time() - t0) * 1000

        passed, detail = case.check(result)
        report.outcomes.append(EvalOutcome(
            case_id=case.case_id, requirement=case.requirement, passed=passed,
            detail=detail, latency_ms=round(latency_ms, 2),
        ))
    return report
