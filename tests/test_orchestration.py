import asyncio
from pathlib import Path

from app.guardrails.policy import GuardrailEngine
from app.orchestration.agent import ComplianceAgent, ResearchAgent
from app.orchestration.fleet import AgentFleet
from app.rag.index import Indexer
from app.rag.ingest import ingest_all

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_indexer() -> Indexer:
    idx = Indexer()
    idx.build(ingest_all(DATA_DIR))
    return idx


def test_research_agent_produces_completed_trace():
    idx = build_indexer()
    agent = ResearchAgent(idx, GuardrailEngine())
    result = agent.run("task-test-1", "wastewater treatment inspection findings")
    assert result.status == "completed"
    assert result.trace["steps"]
    assert any(s["kind"] == "evidence" for s in result.trace["steps"])


def test_research_agent_never_surfaces_secret_docs():
    idx = build_indexer()
    agent = ResearchAgent(idx, GuardrailEngine())  # clearance = CONFIDENTIAL
    result = agent.run("task-test-2", "security incident perimeter breach cyber")
    for step in result.trace["steps"]:
        if step["kind"] == "evidence":
            assert step["detail"]["classification"] != "SECRET"


def test_compliance_agent_can_see_secret_docs():
    idx = build_indexer()
    agent = ComplianceAgent(idx, GuardrailEngine())  # clearance = SECRET
    result = agent.run("task-test-3", "cyber incident perimeter breach")
    classes = {s["detail"].get("classification") for s in result.trace["steps"] if s["kind"] == "evidence"}
    assert classes  # retrieved something
    assert classes <= {"PUBLIC", "CUI", "CONFIDENTIAL", "SECRET"}


def test_fleet_processes_submitted_tasks():
    async def run():
        idx = build_indexer()
        fleet = AgentFleet(idx, GuardrailEngine(), num_workers=2)
        await fleet.start()
        task = fleet.submit("research_agent", "grant reporting delinquent")
        for _ in range(50):
            await asyncio.sleep(0.05)
            if fleet.get(task.task_id).state.value != "queued":
                break
        await fleet.stop()
        return fleet.get(task.task_id)

    finished = asyncio.run(run())
    assert finished.state.value in {"completed", "denied", "requires_approval"}
