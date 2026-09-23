"""Multi-layered guardrails for agent actions in a high-stakes environment.

Layers, applied in order:
  1. Input sanitation  - strip/flag prompt-injection patterns in retrieved or user text
  2. PII / classification redaction - mask sensitive spans before they reach a model or a log
  3. Action policy      - allow/deny/require-approval per tool, keyed by agent role + data classification
  4. Rate & scope limits - bound how much an agent can do per task without a human checkpoint
  5. Output validation  - schema + content checks before a result is released to a user

Every check produces a `GuardrailEvent` that the audit layer can attach to a decision trace,
so a denial or a redaction is itself explainable, not a silent drop.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Verdict(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class GuardrailEvent:
    layer: str
    verdict: Verdict
    reason: str
    detail: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class GuardrailViolation(Exception):
    def __init__(self, event: GuardrailEvent):
        super().__init__(f"[{event.layer}] {event.verdict.value}: {event.reason}")
        self.event = event


# ---------------------------------------------------------------------------
# Layer 1: prompt-injection / input sanitation
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|previous|prior) instructions", re.I),
    re.compile(r"you are now (in )?(developer|admin|debug) mode", re.I),
    re.compile(r"disregard (the )?(system prompt|guardrails|policy)", re.I),
    re.compile(r"act as (an? )?unrestricted", re.I),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.I),
]


def scan_for_injection(text: str) -> GuardrailEvent:
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return GuardrailEvent(
                layer="input_sanitation",
                verdict=Verdict.DENY,
                reason=f"possible prompt injection matched pattern: {pat.pattern}",
            )
    return GuardrailEvent(layer="input_sanitation", verdict=Verdict.ALLOW, reason="clean")


# ---------------------------------------------------------------------------
# Layer 2: PII / classification redaction
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "case_id": re.compile(r"\bCASE-\d{6,}\b"),
}

CLASSIFICATION_MARKERS = ["TOP SECRET", "SECRET", "CONFIDENTIAL", "CUI", "FOUO"]


def redact_pii(text: str) -> tuple[str, GuardrailEvent]:
    redacted = text
    hits: dict[str, int] = {}
    for label, pat in _PII_PATTERNS.items():
        redacted, n = pat.subn(f"[REDACTED:{label.upper()}]", redacted)
        if n:
            hits[label] = n
    verdict = Verdict.REDACT if hits else Verdict.ALLOW
    return redacted, GuardrailEvent(
        layer="pii_redaction", verdict=verdict, reason=f"{sum(hits.values())} spans redacted", detail=hits
    )


def classification_check(text: str, agent_clearance: str) -> GuardrailEvent:
    levels = {"PUBLIC": 0, "CUI": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP SECRET": 4}
    found = [m for m in CLASSIFICATION_MARKERS if m in text.upper()]
    doc_level = max((levels.get(m, 0) for m in found), default=0)
    agent_level = levels.get(agent_clearance.upper(), 0)
    if doc_level > agent_level:
        return GuardrailEvent(
            layer="classification",
            verdict=Verdict.DENY,
            reason=f"document classification exceeds agent clearance ({agent_clearance})",
            detail={"markers": found},
        )
    return GuardrailEvent(layer="classification", verdict=Verdict.ALLOW, reason="within clearance")


# ---------------------------------------------------------------------------
# Layer 3: action policy (RBAC-ish allow/deny/approval per tool)
# ---------------------------------------------------------------------------

@dataclass
class ActionPolicy:
    # role -> set of tool names it may call outright
    allowed: dict[str, set[str]]
    # role -> set of tool names that require human approval
    requires_approval: dict[str, set[str]]

    def check(self, role: str, tool: str) -> GuardrailEvent:
        if tool in self.requires_approval.get(role, set()):
            return GuardrailEvent(
                layer="action_policy", verdict=Verdict.REQUIRE_APPROVAL,
                reason=f"role '{role}' requires human sign-off for tool '{tool}'",
            )
        if tool in self.allowed.get(role, set()):
            return GuardrailEvent(layer="action_policy", verdict=Verdict.ALLOW, reason="permitted")
        return GuardrailEvent(
            layer="action_policy", verdict=Verdict.DENY,
            reason=f"role '{role}' is not permitted to call tool '{tool}'",
        )


DEFAULT_POLICY = ActionPolicy(
    allowed={
        "research_agent": {"search_docs", "read_doc", "summarize"},
        "compliance_agent": {"search_docs", "read_doc", "flag_anomaly"},
        "coordinator_agent": {"search_docs", "dispatch_task", "read_trace"},
    },
    requires_approval={
        "research_agent": {"export_report"},
        "compliance_agent": {"export_report", "close_case"},
        "coordinator_agent": {"terminate_task", "export_report"},
    },
)


# ---------------------------------------------------------------------------
# Layer 4: rate / scope limiting per task (bounds a runaway long-horizon agent)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, max_actions_per_task: int = 40, max_tool_calls_per_minute: int = 20):
        self.max_actions_per_task = max_actions_per_task
        self.max_tool_calls_per_minute = max_tool_calls_per_minute
        self._task_counts: dict[str, int] = {}
        self._minute_window: dict[str, list[float]] = {}

    def check(self, task_id: str) -> GuardrailEvent:
        now = time.time()
        self._task_counts[task_id] = self._task_counts.get(task_id, 0) + 1
        window = [t for t in self._minute_window.get(task_id, []) if now - t < 60]
        window.append(now)
        self._minute_window[task_id] = window

        if self._task_counts[task_id] > self.max_actions_per_task:
            return GuardrailEvent(
                layer="rate_limit", verdict=Verdict.DENY,
                reason=f"task exceeded {self.max_actions_per_task} actions; escalate to human",
                detail={"count": self._task_counts[task_id]},
            )
        if len(window) > self.max_tool_calls_per_minute:
            return GuardrailEvent(
                layer="rate_limit", verdict=Verdict.DENY,
                reason=f"task exceeded {self.max_tool_calls_per_minute} tool calls/minute",
                detail={"count_in_window": len(window)},
            )
        return GuardrailEvent(layer="rate_limit", verdict=Verdict.ALLOW, reason="within limits")


# ---------------------------------------------------------------------------
# Layer 5: output validation
# ---------------------------------------------------------------------------

def validate_output(payload: dict, required_fields: list[str]) -> GuardrailEvent:
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return GuardrailEvent(
            layer="output_validation", verdict=Verdict.DENY,
            reason=f"missing required fields: {missing}",
        )
    return GuardrailEvent(layer="output_validation", verdict=Verdict.ALLOW, reason="schema ok")


# ---------------------------------------------------------------------------
# Orchestrating engine that runs all layers and returns the sanitized result
# ---------------------------------------------------------------------------

class GuardrailEngine:
    def __init__(self, policy: ActionPolicy = DEFAULT_POLICY, rate_limiter: RateLimiter | None = None):
        self.policy = policy
        self.rate_limiter = rate_limiter or RateLimiter()

    def check_input(self, text: str, agent_clearance: str = "CUI") -> tuple[str, list[GuardrailEvent]]:
        events = []
        inj = scan_for_injection(text)
        events.append(inj)
        if inj.verdict == Verdict.DENY:
            raise GuardrailViolation(inj)

        cls = classification_check(text, agent_clearance)
        events.append(cls)
        if cls.verdict == Verdict.DENY:
            raise GuardrailViolation(cls)

        redacted, pii_evt = redact_pii(text)
        events.append(pii_evt)
        return redacted, events

    def check_action(self, role: str, tool: str, task_id: str) -> list[GuardrailEvent]:
        events = [self.policy.check(role, tool), self.rate_limiter.check(task_id)]
        for e in events:
            if e.verdict in (Verdict.DENY, Verdict.REQUIRE_APPROVAL):
                raise GuardrailViolation(e)
        return events

    def check_output(self, payload: dict, required_fields: list[str]) -> list[GuardrailEvent]:
        evt = validate_output(payload, required_fields)
        if evt.verdict == Verdict.DENY:
            raise GuardrailViolation(evt)
        return [evt]
