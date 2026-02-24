# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger("[ METRICS ]")


# Stores latency stats for a single phase
@dataclass
class PhaseMetric:
    name: str
    total_ms: float = 0.0
    count: int = 0
    min_ms: float = float("inf")
    max_ms: float = 0.0


    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


    def record(self, duration_ms: float) -> None:
        self.total_ms += duration_ms
        self.count += 1
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


# Track latency of each pipeline phase for performance monitoring
class PerformanceMetrics:

    def __init__(self) -> None:
        self._phases: dict[str, PhaseMetric] = {}
        self._pipeline_start: float = 0.0


    # Context manager to measure a phase duration
    @contextmanager
    def measure(self, phase_name: str):
        start = time.monotonic()
        try:
            yield
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            if phase_name not in self._phases:
                self._phases[phase_name] = PhaseMetric(name=phase_name)
            self._phases[phase_name].record(duration_ms)
            logger.debug("[PERF] %s: %.0fms", phase_name, duration_ms)


    # Mark the start of a full wake-to-response pipeline
    def start_pipeline(self) -> None:
        self._pipeline_start = time.monotonic()


    # Mark the end of a full pipeline and log total latency
    def end_pipeline(self) -> None:
        if self._pipeline_start > 0:
            total_ms = (time.monotonic() - self._pipeline_start) * 1000
            logger.debug("[PERF] pipeline_total: %.0fms", total_ms)
            if "pipeline_total" not in self._phases:
                self._phases["pipeline_total"] = PhaseMetric(name="pipeline_total")
            self._phases["pipeline_total"].record(total_ms)
            self._pipeline_start = 0.0


    # Return summary stats for all phases
    def get_summary(self) -> dict[str, dict]:
        return {
            name: {
                "avg_ms": round(m.avg_ms, 1),
                "min_ms": round(m.min_ms, 1),
                "max_ms": round(m.max_ms, 1),
                "count": m.count,
            }
            for name, m in self._phases.items()
        }


    # Log a formatted summary of all phase metrics
    def log_summary(self) -> None:
        if not self._phases:
            return
        logger.debug("[PERF] Performance Summary")
        for name, m in sorted(self._phases.items()):
            logger.debug("[PERF]   %s: avg=%.0fms, min=%.0fms, max=%.0fms, count=%d", name, m.avg_ms, m.min_ms, m.max_ms, m.count)


    # Clear all collected metrics
    def reset(self) -> None:
        self._phases.clear()
        self._pipeline_start = 0.0
