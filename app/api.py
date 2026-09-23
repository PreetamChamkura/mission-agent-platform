"""FastAPI application wiring guardrails, RAG, orchestration, anomaly detection,
eval, and audit into one full-stack service for analysts and operators.
"""
from __future__ import annotations

import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db
from app.anomaly.detector import DETECTOR
from app.audit.trace import TRACE_STORE
from app.eval.harness import run_eval_suite
from app.guardrails.policy import GuardrailEngine
from app.orchestration.fleet import AgentFleet
from app.rag.ingest import ingest_all
from app.rag.index import Indexer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Mission Agent Platform", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

indexer = Indexer()
guardrails = GuardrailEngine()
fleet = AgentFleet(indexer, guardrails, num_workers=4)

SEED_METRICS = {
    "permits_filed_per_day": [8, 9, 7, 10, 8, 9, 11, 8, 7, 9],
    "grant_dollars_disbursed_k": [420, 410, 435, 400, 415, 430, 405, 420, 418, 425],
    "incident_reports_per_day": [1, 0, 1, 2, 1, 0, 1, 1, 0, 1],
}


@app.on_event("startup")
async def startup() -> None:
    docs = ingest_all(DATA_DIR)
    indexer.build(docs)
    for metric, values in SEED_METRICS.items():
        for v in values:
            DETECTOR.ingest(metric, v)
    await fleet.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await fleet.stop()


# --- schemas ---------------------------------------------------------------

class TaskRequest(BaseModel):
    agent_role: str
    query: str


class MetricRequest(BaseModel):
    metric: str
    value: float


# --- system / stats ---------------------------------------------------------

@app.get("/api/stats")
def get_stats():
    return {
        "rag_index": indexer.stats(),
        "fleet": fleet.stats(),
        "agent_roles": list(fleet._agents.keys()),
    }


# --- RAG explore -------------------------------------------------------------

@app.get("/api/search")
def search(q: str, top_k: int = 5, max_classification: str = "SECRET"):
    levels = {"PUBLIC": 0, "CUI": 1, "CONFIDENTIAL": 2, "SECRET": 3, "TOP SECRET": 4}
    hits = indexer.search(q, top_k=top_k, max_classification_level=levels.get(max_classification.upper(), 3))
    return {
        "query": q,
        "results": [
            {
                "doc_id": h.doc.doc_id, "score": round(h.score, 4),
                "dense_score": round(h.dense_score, 4), "sparse_score": round(h.sparse_score, 4),
                "title": h.doc.title, "source_system": h.doc.source_system, "record_type": h.doc.record_type,
                "classification": h.doc.classification, "text": h.doc.text[:400],
            }
            for h in hits
        ],
    }


# --- orchestration / fleet ---------------------------------------------------

@app.post("/api/tasks")
def submit_task(req: TaskRequest):
    try:
        task = fleet.submit(req.agent_role, req.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"task_id": task.task_id, "state": task.state.value}


@app.get("/api/tasks")
def list_tasks():
    return [
        {
            "task_id": t.task_id, "agent_role": t.agent_role, "query": t.query,
            "state": t.state.value, "submitted_at": t.submitted_at, "finished_at": t.finished_at,
        }
        for t in fleet.list_tasks()
    ]


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = fleet.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task.task_id, "agent_role": task.agent_role, "query": task.query,
        "state": task.state.value, "error": task.error,
        "result": task.result.answer if task.result else None,
        "trace": task.result.trace if task.result else None,
    }


# --- decision traces (explainability) ---------------------------------------

@app.get("/api/traces")
def list_traces():
    traces = TRACE_STORE.all()
    for t in traces:
        db.persist_trace(t.to_dict())
    return [t.to_dict() for t in traces]


@app.get("/api/traces/{task_id}")
def get_trace(task_id: str):
    trace = TRACE_STORE.get(task_id)
    if not trace:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace.to_dict()


# --- anomaly detection / alerting --------------------------------------------

@app.get("/api/anomalies")
def get_anomalies(limit: int = 50):
    alerts = DETECTOR.recent_alerts(limit)
    return [
        {
            "alert_id": a.alert_id, "metric": a.metric, "value": a.value,
            "baseline_mean": round(a.baseline_mean, 2), "z_score": round(a.z_score, 2),
            "iqr_lower": round(a.iqr_lower, 2), "iqr_upper": round(a.iqr_upper, 2),
            "method": a.method, "severity": a.severity, "message": a.message, "ts": a.ts,
        }
        for a in alerts
    ]


@app.get("/api/metrics")
def get_metric_baselines():
    return [DETECTOR.baseline(m) for m in SEED_METRICS]


@app.post("/api/metrics/ingest")
def ingest_metric(req: MetricRequest):
    alert = DETECTOR.ingest(req.metric, req.value)
    if alert:
        db.persist_alert({
            "metric": alert.metric, "severity": alert.severity, "message": alert.message, "ts": alert.ts,
        })
    return {
        "baseline": DETECTOR.baseline(req.metric),
        "alert": None if not alert else {
            "alert_id": alert.alert_id, "severity": alert.severity, "message": alert.message,
        },
    }


@app.post("/api/metrics/simulate")
def simulate_metric_stream():
    """Push one synthetic, occasionally-anomalous reading per known metric (demo helper)."""
    results = []
    for metric in SEED_METRICS:
        baseline = DETECTOR.baseline(metric)
        mean = baseline.get("mean", 10)
        spike = random.random() < 0.2
        value = mean * random.uniform(2.2, 3.0) if spike else mean + random.uniform(-1, 1)
        alert = DETECTOR.ingest(metric, round(value, 2))
        if alert:
            db.persist_alert({
                "metric": alert.metric, "severity": alert.severity, "message": alert.message, "ts": alert.ts,
            })
        results.append({"metric": metric, "value": round(value, 2), "alert": alert.message if alert else None})
    return results


# --- eval / reliability -------------------------------------------------------

@app.get("/api/eval")
def get_eval_report():
    report = run_eval_suite(indexer)
    report_dict = report.to_dict()
    db.persist_eval_report(report_dict)
    return report_dict


# --- frontend -----------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
