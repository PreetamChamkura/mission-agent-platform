"""Streaming anomaly detection over incoming mission data.

Runs two independent checks per rolling window and flags an anomaly if
either fires, since they catch different failure modes:

  - Z-score: (value - mean) / population stdev, thresholded at warn/critical
    sigma. Sensitive to shifts in a roughly-normal metric, but the mean/stdev
    it computes are themselves skewed by the outlier under test.
  - IQR (Tukey's fences): outlier bounds at Q1 - k*IQR / Q3 + k*IQR using
    quartiles of the window. Robust to the outlier skewing its own baseline,
    and doesn't assume a normal distribution - the standard complement to
    z-score for exactly that reason.

A deviation beyond either threshold raises an `Alert` that the API surfaces
to analysts in real time, instead of requiring someone to notice a bad
number in a dashboard days later.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class Alert:
    alert_id: int
    metric: str
    value: float
    baseline_mean: float
    baseline_stdev: float
    z_score: float
    iqr_lower: float
    iqr_upper: float
    method: str  # "zscore" | "iqr" | "zscore+iqr"
    severity: str  # "warning" | "critical"
    message: str
    ts: float = field(default_factory=time.time)


def _iqr_bounds(values: list[float], k: float) -> tuple[float, float]:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


class StreamAnomalyDetector:
    def __init__(
        self,
        window: int = 20,
        warn_z: float = 2.0,
        critical_z: float = 3.0,
        warn_iqr_k: float = 1.5,
        critical_iqr_k: float = 3.0,
    ):
        self.window = window
        self.warn_z = warn_z
        self.critical_z = critical_z
        self.warn_iqr_k = warn_iqr_k
        self.critical_iqr_k = critical_iqr_k
        self._history: dict[str, list[float]] = {}
        self._alerts: list[Alert] = []
        self._alert_seq = 0

    def ingest(self, metric: str, value: float) -> Alert | None:
        hist = self._history.setdefault(metric, [])
        alert = None
        if len(hist) >= 5:
            mean = statistics.mean(hist)
            stdev = statistics.pstdev(hist) or 1e-6
            z = (value - mean) / stdev

            z_severity = None
            if abs(z) >= self.critical_z:
                z_severity = "critical"
            elif abs(z) >= self.warn_z:
                z_severity = "warning"

            warn_lo, warn_hi = _iqr_bounds(hist, self.warn_iqr_k)
            crit_lo, crit_hi = _iqr_bounds(hist, self.critical_iqr_k)
            iqr_severity = None
            if value < crit_lo or value > crit_hi:
                iqr_severity = "critical"
            elif value < warn_lo or value > warn_hi:
                iqr_severity = "warning"

            severity = None
            if z_severity == "critical" or iqr_severity == "critical":
                severity = "critical"
            elif z_severity == "warning" or iqr_severity == "warning":
                severity = "warning"

            if severity:
                methods = []
                if z_severity:
                    methods.append("zscore")
                if iqr_severity:
                    methods.append("iqr")
                method = "+".join(methods)

                self._alert_seq += 1
                direction = "above" if z > 0 else "below"
                alert = Alert(
                    alert_id=self._alert_seq,
                    metric=metric,
                    value=value,
                    baseline_mean=mean,
                    baseline_stdev=stdev,
                    z_score=z,
                    iqr_lower=warn_lo,
                    iqr_upper=warn_hi,
                    method=method,
                    severity=severity,
                    message=f"{metric} is {abs(z):.1f}sigma {direction} its {self.window}-point baseline "
                            f"({value:.1f} vs mean {mean:.1f}) — flagged by {method}",
                )
                self._alerts.append(alert)

        hist.append(value)
        if len(hist) > self.window:
            hist.pop(0)
        return alert

    def recent_alerts(self, limit: int = 50) -> list[Alert]:
        return list(reversed(self._alerts[-limit:]))

    def baseline(self, metric: str) -> dict:
        hist = self._history.get(metric, [])
        if not hist:
            return {"metric": metric, "count": 0}
        out = {
            "metric": metric,
            "count": len(hist),
            "mean": statistics.mean(hist),
            "stdev": statistics.pstdev(hist) if len(hist) > 1 else 0.0,
            "min": min(hist),
            "max": max(hist),
        }
        if len(hist) >= 4:
            q1, q2, q3 = statistics.quantiles(hist, n=4, method="inclusive")
            out.update({"q1": q1, "median": q2, "q3": q3, "iqr": q3 - q1})
        return out


DETECTOR = StreamAnomalyDetector()
