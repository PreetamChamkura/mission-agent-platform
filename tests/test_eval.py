from pathlib import Path

from app.eval.harness import run_eval_suite
from app.rag.index import Indexer
from app.rag.ingest import ingest_all

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_eval_suite_runs_and_scores():
    idx = Indexer()
    idx.build(ingest_all(DATA_DIR))
    report = run_eval_suite(idx)
    assert len(report.outcomes) == 4
    assert 0.0 <= report.pass_rate <= 1.0
    # the injection-blocking and no-secret-leak requirements must never regress
    by_id = {o.case_id: o for o in report.outcomes}
    assert by_id["EVAL-002"].passed
    assert by_id["EVAL-003"].passed
