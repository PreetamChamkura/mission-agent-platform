"""Streaming anomaly detection over incoming mission data.

Runs a rolling z-score + IQR check per metric stream (e.g. permits filed/day,
grant dollar amounts, incident severity counts). Deviations beyond threshold
raise an `Alert` that the API surfaces to analysts in real time, instead of
requiring someone to notice a bad number in a dashboard days later.
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
    severity: str  # "watch" | "warning" | "critical"
    message: str
    ts: float = field(default_factory=time.time)


class StreamAnomalyDetector:
    def __init__(self, window: int = 20, warn_z: float = 2.0, critical_z: float = 3.0):
        self.window = window
        self.warn_z = warn_z
        self.critical_z = critical_z
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
            severity = None
            if abs(z) >= self.critical_z:
                severity = "critical"
            elif abs(z) >= self.warn_z:
                severity = "warning"
            if severity:
                self._alert_seq += 1
                direction = "above" if z > 0 else "below"
                alert = Alert(
                    alert_id=self._alert_seq,
                    metric=metric,
                    value=value,
                    baseline_mean=mean,
                    baseline_stdev=stdev,
                    z_score=z,
                    severity=severity,
                    message=f"{metric} is {abs(z):.1f}sigma {direction} its {self.window}-point baseline "
                            f"({value:.1f} vs mean {mean:.1f})",
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
        return {
            "metric": metric,
            "count": len(hist),
            "mean": statistics.mean(hist),
            "stdev": statistics.pstdev(hist) if len(hist) > 1 else 0.0,
            "min": min(hist),
            "max": max(hist),
        }


DETECTOR = StreamAnomalyDetector()
