"""Tests for SimpleSkill evaluation metrics.

Reference: SimpleSkill Specification 0.1.0 Section 7 (Evaluation / Selection)
"""

import json
import tempfile
from pathlib import Path

from houyi.domain.skill.metrics import (
    CostMetrics,
    LatencyMetrics,
    MetricsCollector,
    MetricsExporter,
    MetricsStore,
    PrivacyMetrics,
    QualityMetrics,
    ReliabilityMetrics,
    SkillMetrics,
)


class TestQualityMetrics:
    """Test QualityMetrics dataclass."""

    def test_default_values(self):
        metrics = QualityMetrics()
        assert metrics.accuracy is None
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1 is None

    def test_to_dict(self):
        metrics = QualityMetrics(accuracy=0.95, f1=0.88)
        data = metrics.to_dict()
        assert data["accuracy"] == 0.95
        assert data["f1"] == 0.88
        assert "precision" not in data  # None values excluded

    def test_custom_metrics(self):
        metrics = QualityMetrics(custom={"bleu": 0.75, "rouge": 0.82})
        data = metrics.to_dict()
        assert data["bleu"] == 0.75
        assert data["rouge"] == 0.82


class TestLatencyMetrics:
    """Test LatencyMetrics dataclass."""

    def test_default_values(self):
        metrics = LatencyMetrics()
        assert metrics.avg_ms == 0.0
        assert metrics.samples == 0

    def test_to_dict(self):
        metrics = LatencyMetrics(
            avg_ms=120.5,
            min_ms=50.0,
            max_ms=500.0,
            p95_ms=350.0,
            samples=100,
        )
        data = metrics.to_dict()
        assert data["avg"] == 120.5
        assert data["min"] == 50.0
        assert data["p95"] == 350.0
        assert data["samples"] == 100


class TestReliabilityMetrics:
    """Test ReliabilityMetrics dataclass."""

    def test_default_values(self):
        metrics = ReliabilityMetrics()
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.total_count == 0

    def test_success_rate(self):
        metrics = ReliabilityMetrics(success_count=80, error_count=20, total_count=100)
        assert metrics.success_rate == 0.8
        assert metrics.error_rate == 0.2

    def test_success_rate_zero_total(self):
        metrics = ReliabilityMetrics()
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 0.0

    def test_timeout_rate(self):
        metrics = ReliabilityMetrics(
            success_count=90,
            timeout_count=5,
            error_count=5,
            total_count=100,
        )
        assert metrics.timeout_rate == 0.05

    def test_to_dict(self):
        metrics = ReliabilityMetrics(
            success_count=95,
            error_count=5,
            total_count=100,
        )
        data = metrics.to_dict()
        assert data["success_count"] == 95
        assert data["success_rate"] == 0.95
        assert data["error_rate"] == 0.05


class TestCostMetrics:
    """Test CostMetrics dataclass."""

    def test_default_values(self):
        metrics = CostMetrics()
        assert metrics.tokens_total == 0
        assert metrics.usd_estimate is None

    def test_to_dict(self):
        metrics = CostMetrics(
            tokens_input=1000,
            tokens_output=500,
            tokens_total=1500,
            api_calls=5,
            usd_estimate=0.02,
        )
        data = metrics.to_dict()
        assert data["tokens_input"] == 1000
        assert data["tokens_output"] == 500
        assert data["tokens_total"] == 1500
        assert data["api_calls"] == 5
        assert data["usd_estimate"] == 0.02


class TestPrivacyMetrics:
    """Test PrivacyMetrics dataclass."""

    def test_default_values(self):
        metrics = PrivacyMetrics()
        assert metrics.local_only is True
        assert metrics.data_egress is False

    def test_to_dict(self):
        metrics = PrivacyMetrics(local_only=False, data_egress=True)
        data = metrics.to_dict()
        assert data["local_only"] is False
        assert data["data_egress"] is True


class TestSkillMetrics:
    """Test SkillMetrics dataclass."""

    def test_creation(self):
        metrics = SkillMetrics(skill_name="test_skill")
        assert metrics.skill_name == "test_skill"
        assert metrics.quality is not None
        assert metrics.latency is not None

    def test_to_dict(self):
        metrics = SkillMetrics(skill_name="test_skill")
        metrics.quality.accuracy = 0.95
        metrics.reliability.success_count = 100
        metrics.reliability.total_count = 100

        data = metrics.to_dict()
        assert data["skill_name"] == "test_skill"
        assert data["quality"]["accuracy"] == 0.95
        assert data["reliability"]["success_count"] == 100

    def test_from_dict(self):
        data = {
            "skill_name": "test_skill",
            "quality": {"accuracy": 0.9},
            "latency": {"avg": 100, "p95": 200, "samples": 50},
            "reliability": {"success_count": 45, "error_count": 5, "total_count": 50},
            "collected_at": "2024-01-01T00:00:00+00:00",
        }
        metrics = SkillMetrics.from_dict(data)
        assert metrics.skill_name == "test_skill"
        assert metrics.quality.accuracy == 0.9
        assert metrics.latency.avg_ms == 100
        assert metrics.reliability.success_count == 45


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_record_latency(self):
        collector = MetricsCollector("test_skill")
        collector.record_latency(100.0)
        collector.record_latency(200.0)
        collector.record_latency(150.0)

        metrics = collector.get_metrics()
        assert metrics.latency.samples == 3
        assert metrics.latency.avg_ms == 150.0
        assert metrics.latency.min_ms == 100.0
        assert metrics.latency.max_ms == 200.0

    def test_record_success_error(self):
        collector = MetricsCollector("test_skill")
        collector.record_success()
        collector.record_success()
        collector.record_error()

        metrics = collector.get_metrics()
        assert metrics.reliability.success_count == 2
        assert metrics.reliability.error_count == 1
        assert metrics.reliability.total_count == 3

    def test_record_timeout(self):
        collector = MetricsCollector("test_skill")
        collector.record_timeout()

        metrics = collector.get_metrics()
        assert metrics.reliability.timeout_count == 1
        assert metrics.reliability.total_count == 1

    def test_record_retry(self):
        collector = MetricsCollector("test_skill")
        collector.record_retry()
        collector.record_retry()

        metrics = collector.get_metrics()
        assert metrics.reliability.retry_count == 2

    def test_record_tokens(self):
        collector = MetricsCollector("test_skill")
        collector.record_tokens(input_tokens=100, output_tokens=50)
        collector.record_tokens(input_tokens=200, output_tokens=100)

        metrics = collector.get_metrics()
        assert metrics.cost.tokens_input == 300
        assert metrics.cost.tokens_output == 150
        assert metrics.cost.tokens_total == 450

    def test_record_api_call(self):
        collector = MetricsCollector("test_skill")
        collector.record_api_call()
        collector.record_api_call()

        metrics = collector.get_metrics()
        assert metrics.cost.api_calls == 2

    def test_set_quality(self):
        collector = MetricsCollector("test_skill")
        collector.set_quality(accuracy=0.95, f1=0.88, custom_metric=0.75)

        metrics = collector.get_metrics()
        assert metrics.quality.accuracy == 0.95
        assert metrics.quality.f1 == 0.88
        assert metrics.quality.custom["custom_metric"] == 0.75

    def test_set_privacy(self):
        collector = MetricsCollector("test_skill")
        collector.set_privacy(local_only=False, data_egress=True)

        metrics = collector.get_metrics()
        assert metrics.privacy.local_only is False
        assert metrics.privacy.data_egress is True

    def test_measure_execution(self):
        collector = MetricsCollector("test_skill")

        with collector.measure_execution():
            # Simulate some work
            _ = sum(range(1000))

        metrics = collector.get_metrics()
        assert metrics.latency.samples == 1
        assert metrics.latency.avg_ms > 0  # Should have recorded some time

    def test_reset(self):
        collector = MetricsCollector("test_skill")
        collector.record_success()
        collector.record_latency(100.0)

        collector.reset()

        metrics = collector.get_metrics()
        assert metrics.reliability.total_count == 0
        assert metrics.latency.samples == 0


class TestMetricsExporter:
    """Test MetricsExporter class."""

    def test_to_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            metrics = SkillMetrics(skill_name="test_skill")
            metrics.quality.accuracy = 0.95

            MetricsExporter.to_json(metrics, path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["skill_name"] == "test_skill"
            assert data["quality"]["accuracy"] == 0.95

    def test_to_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            metrics_list = [
                SkillMetrics(skill_name="skill1"),
                SkillMetrics(skill_name="skill2"),
            ]

            MetricsExporter.to_json_lines(metrics_list, path)

            assert path.exists()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["skill_name"] == "skill1"

    def test_to_json_lines_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            MetricsExporter.to_json_lines([SkillMetrics(skill_name="skill1")], path)
            MetricsExporter.to_json_lines([SkillMetrics(skill_name="skill2")], path, append=True)

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

    def test_to_opentelemetry_attributes(self):
        metrics = SkillMetrics(skill_name="test_skill")
        metrics.latency.avg_ms = 100.0
        metrics.latency.p95_ms = 200.0
        metrics.reliability.success_count = 95
        metrics.reliability.total_count = 100
        metrics.cost.tokens_total = 1500
        metrics.quality.accuracy = 0.95

        attrs = MetricsExporter.to_opentelemetry_attributes(metrics)

        assert attrs["skill.name"] == "test_skill"
        assert attrs["skill.latency.avg_ms"] == 100.0
        assert attrs["skill.latency.p95_ms"] == 200.0
        assert attrs["skill.reliability.success_rate"] == 0.95
        assert attrs["skill.cost.tokens_total"] == 1500
        assert attrs["skill.quality.accuracy"] == 0.95


class TestMetricsStore:
    """Test MetricsStore class."""

    def test_store_and_get_latest(self):
        store = MetricsStore()

        m1 = SkillMetrics(skill_name="test_skill")
        m1.quality.accuracy = 0.9
        store.store(m1)

        m2 = SkillMetrics(skill_name="test_skill")
        m2.quality.accuracy = 0.95
        store.store(m2)

        latest = store.get_latest("test_skill")
        assert latest is not None
        assert latest.quality.accuracy == 0.95

    def test_get_latest_not_found(self):
        store = MetricsStore()
        assert store.get_latest("nonexistent") is None

    def test_get_all(self):
        store = MetricsStore()
        for i in range(3):
            m = SkillMetrics(skill_name="test_skill")
            m.quality.accuracy = 0.9 + i * 0.01
            store.store(m)

        all_metrics = store.get_all("test_skill")
        assert len(all_metrics) == 3

    def test_list_skills(self):
        store = MetricsStore()
        store.store(SkillMetrics(skill_name="skill1"))
        store.store(SkillMetrics(skill_name="skill2"))
        store.store(SkillMetrics(skill_name="skill1"))

        skills = store.list_skills()
        assert set(skills) == {"skill1", "skill2"}

    def test_aggregate(self):
        store = MetricsStore()

        # Store multiple metrics entries
        for _ in range(3):
            m = SkillMetrics(skill_name="test_skill")
            m.reliability.success_count = 10
            m.reliability.error_count = 1
            m.reliability.total_count = 11
            m.cost.tokens_input = 100
            m.cost.tokens_output = 50
            m.cost.tokens_total = 150
            m.cost.api_calls = 1
            store.store(m)

        aggregated = store.aggregate("test_skill")
        assert aggregated is not None
        assert aggregated.reliability.success_count == 30
        assert aggregated.reliability.total_count == 33
        assert aggregated.cost.tokens_total == 450
        assert aggregated.cost.api_calls == 3
        assert aggregated.metadata["aggregated_from"] == 3

    def test_aggregate_not_found(self):
        store = MetricsStore()
        assert store.aggregate("nonexistent") is None

    def test_export_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "all_metrics.json"
            store = MetricsStore()
            store.store(SkillMetrics(skill_name="skill1"))
            store.store(SkillMetrics(skill_name="skill2"))

            store.export_all(path)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "skill1" in data
            assert "skill2" in data
