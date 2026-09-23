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


def test_hits_carry_both_dense_and_sparse_scores():
    idx = build_indexer()
    hits = idx.search("wastewater treatment inspection", top_k=3)
    assert hits
    for h in hits:
        assert isinstance(h.dense_score, float)
        assert isinstance(h.sparse_score, float)
        assert -1.0001 <= h.dense_score <= 1.0001  # cosine similarity range


def test_dense_retrieval_finds_paraphrases_sparse_alone_would_miss():
    idx = build_indexer()
    # no literal token overlap with "unresolved status" wording in the source doc,
    # but semantically it's the same finding as the Northgate fuel storage report
    hits = idx.search("facility that still needs to submit its overdue paperwork", top_k=5)
    assert any("northgate" in h.doc.doc_id.lower() for h in hits)
