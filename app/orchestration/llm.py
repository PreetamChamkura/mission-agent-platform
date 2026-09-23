"""Pluggable reasoning backend.

`MockLLM` is a deterministic, offline stand-in used so the whole platform runs
without network access or API keys - the same reason the RAG index above is
TF-IDF rather than a hosted embedding call. Swap in a real Claude client by
implementing the same `.complete()` signature and passing it into agents.
"""
from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class MockLLM:
    """Cheap extractive 'reasoning': summarizes retrieved evidence deterministically.

    Good enough to drive the orchestration/guardrail/eval demo end-to-end without
    any external dependency, and trivially swappable for a real model client.
    """

    def complete(self, system: str, prompt: str) -> str:
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        evidence_lines = [ln for ln in lines if ln.startswith("- ")]
        if evidence_lines:
            top = evidence_lines[:3]
            return (
                "Based on " + str(len(evidence_lines)) + " retrieved record(s), the most relevant findings are:\n"
                + "\n".join(top)
                + "\nRecommendation: route to a human analyst for final sign-off if any finding involves "
                  "elevated severity or an unresolved status."
            )
        return "No relevant evidence was retrieved for this query; recommend broadening the search scope."
