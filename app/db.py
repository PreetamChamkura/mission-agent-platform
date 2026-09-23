"""Minimal persistence layer (SQLite) for audit durability.

Decision traces and alerts live in memory for hot access during a session
(see app/audit/trace.py, app/anomaly/detector.py) but are mirrored here so
an audit trail survives a process restart, as it must in a compliance-bearing
deployment. Uses only the stdlib sqlite3 driver - no external DB service
required, which matters for the air-gapped deployment target.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "mission_platform.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_traces (
            task_id TEXT PRIMARY KEY,
            agent_role TEXT,
            goal TEXT,
            status TEXT,
            confidence REAL,
            payload_json TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT,
            severity TEXT,
            message TEXT,
            payload_json TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pass_rate REAL,
            payload_json TEXT,
            created_at REAL
        )
    """)
    return conn


def persist_trace(trace_dict: dict) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO decision_traces "
            "(task_id, agent_role, goal, status, confidence, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace_dict["task_id"], trace_dict["agent_role"], trace_dict["goal"],
                trace_dict["status"], trace_dict.get("confidence"),
                json.dumps(trace_dict), trace_dict["started_at"],
            ),
        )
    conn.close()


def persist_alert(alert_dict: dict) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO alerts (metric, severity, message, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (alert_dict["metric"], alert_dict["severity"], alert_dict["message"],
             json.dumps(alert_dict), alert_dict["ts"]),
        )
    conn.close()


def persist_eval_report(report_dict: dict) -> None:
    conn = get_conn()
    with conn:
        conn.execute(
            "INSERT INTO eval_reports (pass_rate, payload_json, created_at) VALUES (?, ?, ?)",
            (report_dict["pass_rate"], json.dumps(report_dict), report_dict["generated_at"]),
        )
    conn.close()
