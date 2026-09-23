"""Ingestion connectors that pull heterogeneous, previously siloed sources into one
unified document store with normalized metadata (source system, classification,
record type). Each connector is small and independent, the way separate agency
systems of record would be onboarded one at a time.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    source_system: str
    record_type: str
    classification: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def ingest_csv(path: Path, source_system: str, record_type: str, classification: str,
                title_field: str, text_fields: list[str]) -> list[Document]:
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = " | ".join(f"{k}: {row.get(k, '')}" for k in text_fields)
            docs.append(Document(
                doc_id=f"{source_system}:{path.stem}:{i}",
                source_system=source_system,
                record_type=record_type,
                classification=classification,
                title=str(row.get(title_field, f"{path.stem}-{i}")),
                text=text,
                metadata=row,
            ))
    return docs


def ingest_json(path: Path, source_system: str, record_type: str, classification: str,
                 title_field: str, text_fields: list[str]) -> list[Document]:
    docs = []
    records = json.loads(path.read_text(encoding="utf-8"))
    for i, row in enumerate(records):
        text = " | ".join(f"{k}: {row.get(k, '')}" for k in text_fields)
        docs.append(Document(
            doc_id=f"{source_system}:{path.stem}:{i}",
            source_system=source_system,
            record_type=record_type,
            classification=classification,
            title=str(row.get(title_field, f"{path.stem}-{i}")),
            text=text,
            metadata=row,
        ))
    return docs


def ingest_text_dir(path: Path, source_system: str, record_type: str, classification: str) -> list[Document]:
    docs = []
    for i, fp in enumerate(sorted(path.glob("*.txt"))):
        docs.append(Document(
            doc_id=f"{source_system}:{fp.stem}",
            source_system=source_system,
            record_type=record_type,
            classification=classification,
            title=fp.stem.replace("_", " ").title(),
            text=fp.read_text(encoding="utf-8"),
            metadata={"filename": fp.name},
        ))
    return docs


def ingest_all(data_dir: Path) -> list[Document]:
    """Wire up every connector for the demo's synthetic siloed sources."""
    docs: list[Document] = []
    docs += ingest_csv(
        data_dir / "permits.csv", source_system="PermitsDB", record_type="permit",
        classification="CUI", title_field="permit_id",
        text_fields=["permit_id", "applicant", "facility_type", "county", "status", "notes"],
    )
    docs += ingest_json(
        data_dir / "grants.json", source_system="GrantsSystem", record_type="grant",
        classification="CUI", title_field="grant_id",
        text_fields=["grant_id", "recipient", "program", "amount_usd", "fiscal_year", "risk_notes"],
    )
    docs += ingest_text_dir(
        data_dir / "inspection_reports", source_system="InspectionArchive",
        record_type="inspection_report", classification="CONFIDENTIAL",
    )
    docs += ingest_csv(
        data_dir / "incidents.csv", source_system="IncidentTracker", record_type="incident",
        classification="SECRET", title_field="incident_id",
        text_fields=["incident_id", "category", "location", "severity", "summary"],
    )
    return docs
