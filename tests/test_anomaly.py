from app.anomaly.detector import StreamAnomalyDetector


def test_no_alert_within_stable_baseline():
    det = StreamAnomalyDetector(window=20)
    alert = None
    for v in [10, 11, 9, 10, 10, 11, 9, 10]:
        alert = det.ingest("metric_a", v)
    assert alert is None


def test_zscore_flags_a_spike():
    det = StreamAnomalyDetector(window=20, warn_z=2.0, critical_z=3.0)
    for v in [10, 10, 11, 9, 10, 10, 9, 11, 10]:
        det.ingest("metric_b", v)
    alert = det.ingest("metric_b", 40)
    assert alert is not None
    assert "zscore" in alert.method
    assert alert.severity in {"warning", "critical"}


def test_iqr_flags_an_outlier_that_skews_its_own_mean():
    # a single huge value drags the mean/stdev toward itself, which can mask
    # it from a pure z-score check; IQR (quartile-based, robust to the
    # outlier under test) should still catch it via its bounds.
    det = StreamAnomalyDetector(window=10, warn_iqr_k=1.5, critical_iqr_k=3.0)
    for v in [5, 5, 6, 5, 4, 5]:
        det.ingest("metric_c", v)
    alert = det.ingest("metric_c", 50)
    assert alert is not None
    assert "iqr" in alert.method


def test_baseline_reports_quartiles_once_enough_data():
    det = StreamAnomalyDetector(window=20)
    for v in [1, 2, 3, 4, 5]:
        det.ingest("metric_d", v)
    baseline = det.baseline("metric_d")
    assert "q1" in baseline and "q3" in baseline and "iqr" in baseline
