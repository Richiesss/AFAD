"""Tests for MetricsCollector."""

import time

from src.utils.metrics import MetricsCollector, RoundMetrics


class TestRoundMetrics:
    def test_defaults(self):
        m = RoundMetrics(round_num=1)
        assert m.round_num == 1
        assert m.loss == 0.0
        assert m.accuracy == 0.0
        assert m.num_clients == 0
        assert m.wall_time == 0.0


class TestMetricsCollector:
    def test_record_and_get_latest(self):
        mc = MetricsCollector()
        mc.start_round()
        result = mc.record_round(round_num=1, loss=0.5, accuracy=0.8, num_clients=3)
        assert result.round_num == 1
        assert result.accuracy == 0.8
        assert mc.get_latest() is result

    def test_get_latest_empty(self):
        mc = MetricsCollector()
        assert mc.get_latest() is None

    def test_summary(self):
        mc = MetricsCollector()
        mc.start_round()
        mc.record_round(round_num=1, loss=0.5, accuracy=0.7, num_clients=5)
        mc.start_round()
        mc.record_round(round_num=2, loss=0.3, accuracy=0.9, num_clients=5)

        s = mc.summary()
        assert s["num_rounds"] == 2
        assert s["best_accuracy"] == 0.9
        assert s["final_accuracy"] == 0.9
        assert s["best_loss"] == 0.3
        assert s["final_loss"] == 0.3

    def test_summary_empty(self):
        mc = MetricsCollector()
        assert mc.summary() == {}

    def test_wall_time_measured(self):
        mc = MetricsCollector()
        mc.start_round()
        time.sleep(0.01)
        result = mc.record_round(round_num=1)
        assert result.wall_time > 0.0
