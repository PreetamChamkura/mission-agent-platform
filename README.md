# Mission Agent Platform

A demo full-stack platform for running AI agents safely over heterogeneous,
siloed mission data in a high-stakes federal environment. It's a working,
runnable reference architecture, not a slide deck: every requirement below
maps to real code you can read, run, and test.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py            # http://localhost:8000
pytest tests/ -q           # 19 tests covering every layer
```

## What it covers

| Requirement | Where |
|---|---|
| Multi-layered guardrails | [`app/guardrails/policy.py`](app/guardrails/policy.py) — 5 layers: prompt-injection scanning, PII redaction, classification-based access control, per-task rate limiting, output schema validation |
| RAG over large, heterogeneous federal datasets | [`app/rag/ingest.py`](app/rag/ingest.py) (4 independent connectors: CSV, JSON, text archive) + [`app/rag/index.py`](app/rag/index.py) (hybrid dense/sparse retrieval — local `BAAI/bge-small-en-v1.5` embeddings via `fastembed`/ONNX in a FAISS `IndexFlatIP` vector store, fused with a hand-rolled TF-IDF sparse index — with classification/source/type filters applied pre-scoring, no external embedding API) |
| Orchestration for fleets of async, long-horizon agents | [`app/orchestration/fleet.py`](app/orchestration/fleet.py) — bounded asyncio worker pool, custom task state machine, human-checkpointed approval flow |
| Automatic alerting on data anomalies | [`app/anomaly/detector.py`](app/anomaly/detector.py) — streaming anomaly detector combining rolling z-score and IQR/Tukey's-fences checks over ingested metric streams |
| Interfaces that show how an agent reached a decision | [`app/audit/trace.py`](app/audit/trace.py) + the **Decision Traces** tab in the UI — every guardrail check, retrieved document, and reasoning step is logged and rendered |
| Pipelines that make siloed data accessible to agents | `app/rag/ingest.py` connectors onboard 4 independent synthetic "agency systems" (`PermitsDB`, `GrantsSystem`, `InspectionArchive`, `IncidentTracker`) into one unified, filterable index |
| Evaluation infra measuring reliability against mission requirements | [`app/eval/`](app/eval/) — each eval case encodes an actual mission requirement ("never leak SECRET data to an under-cleared agent") with an automatic checker, not a generic benchmark |
| Full-stack tooling for analysts to query/visualize/explore | [`frontend/index.html`](frontend/index.html) — single-page dashboard: query agents, search evidence directly, watch the fleet, inspect traces, watch alerts, run evals |
| Deploy into secure, air-gapped, cloud-native environments | [`deploy/`](deploy/) — hardened multi-stage Dockerfile (non-root, read-only rootfs, dropped capabilities), Kubernetes manifests with a default-deny egress `NetworkPolicy`, and [`deploy/AIRGAP.md`](deploy/AIRGAP.md) for offline builds |

## Architecture

```
                    ┌─────────────────────────────┐
  analyst  ───────► │   frontend/index.html        │
                    │   (query, explore, monitor)  │
                    └──────────────┬───────────────┘
                                   │ REST (app/api.py)
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐        ┌──────────────────┐        ┌────────────────┐
│ AgentFleet     │        │ Indexer (RAG)     │        │ AnomalyDetector │
│ async worker   │◄──────►│ FAISS dense +     │        │ z-score + IQR   │
│ pool + task    │  read  │ TF-IDF sparse,    │        │ alerts on data  │
│ state machine  │        │ 4 siloed sources  │        │ streams         │
└───────┬────────┘        └──────────────────┘        └────────────────┘
        │ every step
        ▼
┌────────────────┐      ┌──────────────────┐
│ GuardrailEngine │      │ DecisionTrace /   │
│ 5-layer checks  │─────►│ TraceStore + DB    │  ← explainability / audit
└────────────────┘      └──────────────────┘

Eval harness (app/eval/) runs the same agents against mission-requirement
test cases and scores pass/fail + latency — meant to gate any policy/prompt/
model change before it ships.
```

## Why the design choices

- **No external LLM or embedding API calls anywhere.** Reasoning is a mock
  client behind an `LLMClient` interface (`app/orchestration/llm.py`), and
  retrieval runs a local embedding model (`app/rag/embeddings.py`) into an
  in-process FAISS index (`app/rag/vector_store.py`) — no hosted inference
  call for either. That's the actual constraint in an air-gapped enclave;
  swap `MockLLM` for a real client behind the same interface once one is
  available, and see `deploy/AIRGAP.md` for baking the embedding model into
  the image so it never needs runtime network access.
- **Guardrails run on every agent, every step**, not as an afterthought:
  input sanitation and PII redaction before retrieval, classification checks
  before results are used, an action policy before any tool call, rate limits
  bounding a runaway long-horizon task, and output validation before release.
- **Every guardrail decision and retrieval is logged to a `DecisionTrace`**
  so a denial, redaction, or low-confidence answer is itself auditable —
  this is what the "how did the agent decide that" interface is built on.
- **Data classification is enforced at the retrieval layer, not just the
  UI**: a `research_agent` (CONFIDENTIAL clearance) structurally cannot
  retrieve `SECRET` documents, verified by `tests/test_orchestration.py` and
  `EVAL-003` in the reliability suite.

## Extending it

- Real model: implement `LLMClient.complete()` in `app/orchestration/llm.py`
  against your enclave-approved model server.
- Larger corpus: `app/rag/vector_store.py` uses FAISS `IndexFlatIP` (exact
  search), the right choice at this corpus size; past a few hundred thousand
  vectors, swap in `IndexIVFFlat` or `IndexHNSW` without touching the caller.
- Different embedding model: swap the model name in `app/rag/embeddings.py`;
  the metadata filters (classification, source system, record type) and the
  dense/sparse fusion in `Indexer.search` carry over unchanged.
- Real data sources: add a connector in `app/rag/ingest.py` following the
  existing CSV/JSON/text-dir pattern — each source stays independent.
- Real audit sink: point `app/db.py` at your approved long-term log store
  instead of local SQLite.
