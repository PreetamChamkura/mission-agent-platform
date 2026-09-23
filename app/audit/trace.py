"""Decision-trace recording for explainability.

Every step an agent takes (thought, tool call, retrieved evidence, guardrail
event, final answer) is appended to a `DecisionTrace`. The trace is what an
operator audits: not just the final output, but *why* the agent produced it -
which documents it pulled, which guardrails fired, and how confident it was.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

_step_counter = itertools.count(1)


@dataclass
class TraceStep:
    step_id: int
    kind: str  # "thought" | "tool_call" | "evidence" | "guardrail" | "answer"
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class DecisionTrace:
    task_id: str
    agent_role: str
    goal: str
    steps: list[TraceStep] = field(default_factory=list)
    status: str = "running"  # running | completed | denied | failed
    confidence: float | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add(self, kind: str, summary: str, **detail) -> TraceStep:
        step = TraceStep(step_id=next(_step_counter), kind=kind, summary=summary, detail=detail)
        self.steps.append(step)
        return step

    def finish(self, status: str, confidence: float | None = None) -> None:
        self.status = status
        self.confidence = confidence
        self.finished_at = time.time()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "goal": self.goal,
            "status": self.status,
            "confidence": self.confidence,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (self.finished_at - self.started_at) if self.finished_at else None,
            "steps": [
                {"step_id": s.step_id, "kind": s.kind, "summary": s.summary, "detail": s.detail, "ts": s.ts}
                for s in self.steps
            ],
        }


class TraceStore:
    """In-memory store of every decision trace produced, keyed by task_id."""

    def __init__(self):
        self._traces: dict[str, DecisionTrace] = {}

    def create(self, task_id: str, agent_role: str, goal: str) -> DecisionTrace:
        trace = DecisionTrace(task_id=task_id, agent_role=agent_role, goal=goal)
        self._traces[task_id] = trace
        return trace

    def get(self, task_id: str) -> DecisionTrace | None:
        return self._traces.get(task_id)

    def all(self) -> list[DecisionTrace]:
        return sorted(self._traces.values(), key=lambda t: t.started_at, reverse=True)


TRACE_STORE = TraceStore()
