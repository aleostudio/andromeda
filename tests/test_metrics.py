# Copyright (c) 2026 Alessandro Orrù
# Licensed under MIT

import time
import pytest
from andromeda.metrics import PerformanceMetrics, PhaseMetric


class TestPhaseMetric:
    def test_initial_values(self):
        m = PhaseMetric(name="test")
        assert m.count == 0
        assert m.total_ms == 0.0
        assert m.min_ms == float("inf")
        assert m.max_ms == 0.0

    def test_avg_zero_count(self):
        m = PhaseMetric(name="test")
        assert m.avg_ms == 0.0

    def test_record_single(self):
        m = PhaseMetric(name="test")
        m.record(100.0)
        assert m.count == 1
        assert m.total_ms == 100.0
        assert m.min_ms == 100.0
        assert m.max_ms == 100.0
        assert m.avg_ms == 100.0

    def test_record_multiple(self):
        m = PhaseMetric(name="test")
        m.record(100.0)
        m.record(200.0)
        m.record(300.0)
        assert m.count == 3
        assert m.total_ms == 600.0
        assert m.min_ms == 100.0
        assert m.max_ms == 300.0
        assert m.avg_ms == pytest.approx(200.0)


class TestPerformanceMetrics:
    def test_measure_records_phase(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stt"):
            time.sleep(0.01)

        summary = metrics.get_summary()
        assert "stt" in summary
        assert summary["stt"]["count"] == 1
        assert summary["stt"]["avg_ms"] > 0

    def test_measure_multiple_phases(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stt"):
            pass
        with metrics.measure("tts"):
            pass

        summary = metrics.get_summary()
        assert "stt" in summary
        assert "tts" in summary

    def test_measure_same_phase_accumulates(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stt"):
            pass
        with metrics.measure("stt"):
            pass

        summary = metrics.get_summary()
        assert summary["stt"]["count"] == 2

    def test_pipeline_tracking(self):
        metrics = PerformanceMetrics()
        metrics.start_pipeline()
        time.sleep(0.01)
        metrics.end_pipeline()

        summary = metrics.get_summary()
        assert "pipeline_total" in summary
        assert summary["pipeline_total"]["count"] == 1
        assert summary["pipeline_total"]["avg_ms"] > 0

    def test_end_pipeline_without_start(self):
        metrics = PerformanceMetrics()
        metrics.end_pipeline()  # Should not crash
        assert "pipeline_total" not in metrics.get_summary()

    def test_get_summary_empty(self):
        metrics = PerformanceMetrics()
        assert metrics.get_summary() == {}

    def test_log_summary_empty(self):
        metrics = PerformanceMetrics()
        metrics.log_summary()  # Should not crash

    def test_log_summary_with_data(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stt"):
            pass
        metrics.log_summary()  # Should not crash

    def test_reset(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stt"):
            pass
        metrics.start_pipeline()
        metrics.end_pipeline()

        metrics.reset()
        assert metrics.get_summary() == {}

    def test_measure_exception_still_records(self):
        metrics = PerformanceMetrics()
        with pytest.raises(ValueError):
            with metrics.measure("failing"):
                raise ValueError("test error")

        summary = metrics.get_summary()
        assert "failing" in summary
        assert summary["failing"]["count"] == 1

    def test_summary_format(self):
        metrics = PerformanceMetrics()
        with metrics.measure("stt"):
            time.sleep(0.01)

        summary = metrics.get_summary()
        stt = summary["stt"]
        assert "avg_ms" in stt
        assert "min_ms" in stt
        assert "max_ms" in stt
        assert "count" in stt
        assert isinstance(stt["avg_ms"], float)
        assert isinstance(stt["count"], int)
