from pathlib import Path

from app.rag.index import Indexer
from app.rag.ingest import ingest_all

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_indexer() -> Indexer:
    idx = Indexer()
    idx.build(ingest_all(DATA_DIR))
    return idx


def test_ingests_all_sources():
    idx = build_indexer()
    stats = idx.stats()
    assert stats["total_documents"] > 0
    assert set(stats["by_source_system"]) == {
        "PermitsDB", "GrantsSystem", "InspectionArchive", "IncidentTracker",
    }


def test_search_returns_relevant_hits():
    idx = build_indexer()
    hits = idx.search("fuel storage tank integrity overdue", top_k=3)
    assert hits
    assert any("northgate" in h.doc.doc_id.lower() or "fuel" in h.doc.text.lower() for h in hits)


def test_classification_filter_excludes_secret():
    idx = build_indexer()
    hits = idx.search("security incident facility", top_k=10, max_classification_level=1)  # CUI only
    assert all(h.doc.classification in {"PUBLIC", "CUI"} for h in hits)
