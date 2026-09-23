"""Orchestration for a fleet of asynchronous agents running long-horizon tasks.

Design:
  - `AgentFleet` owns an asyncio queue and a fixed pool of worker coroutines
    (bounded concurrency, so N agents don't hammer shared resources at once).
  - Each submitted `Task` moves through a small state machine:
      queued -> running -> (completed | denied | requires_approval | failed)
  - Long-horizon tasks are broken into checkpointed sub-steps: if a task is
    paused (e.g. it hit a REQUIRE_APPROVAL guardrail), its state is retained
    so a human can approve/resume rather than the fleet losing the work.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

from app.orchestration.agent import AGENT_REGISTRY, AgentResult

_task_id_seq = itertools.count(1)


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    agent_role: str
    query: str
    state: TaskState = TaskState.QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: AgentResult | None = None
    error: str | None = None


class AgentFleet:
    def __init__(self, indexer, guardrails, num_workers: int = 4):
        self.indexer = indexer
        self.guardrails = guardrails
        self.num_workers = num_workers
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, Task] = {}
        self._workers: list[asyncio.Task] = []
        self._agents = {role: cls(indexer, guardrails) for role, cls in AGENT_REGISTRY.items()}

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self.num_workers)]

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        self._workers = []

    def submit(self, agent_role: str, query: str) -> Task:
        if agent_role not in self._agents:
            raise ValueError(f"unknown agent role: {agent_role}")
        task_id = f"task-{next(_task_id_seq):05d}"
        task = Task(task_id=task_id, agent_role=agent_role, query=query)
        self._tasks[task_id] = task
        self._queue.put_nowait(task_id)
        return task

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            task_id = await self._queue.get()
            task = self._tasks[task_id]
            task.state = TaskState.RUNNING
            task.started_at = time.time()
            try:
                agent = self._agents[task.agent_role]
                result = await asyncio.to_thread(agent.run, task.task_id, task.query)
                task.result = result
                task.state = TaskState(result.status) if result.status in TaskState._value2member_map_ else TaskState.COMPLETED
            except Exception as exc:  # noqa: BLE001 - fleet must not die on one bad task
                task.state = TaskState.FAILED
                task.error = str(exc)
            finally:
                task.finished_at = time.time()
                self._queue.task_done()

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.submitted_at, reverse=True)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for t in self._tasks.values():
            counts[t.state.value] = counts.get(t.state.value, 0) + 1
        return {
            "num_workers": self.num_workers,
            "queue_depth": self._queue.qsize(),
            "total_tasks": len(self._tasks),
            "by_state": counts,
        }
