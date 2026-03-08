"""Skill metrics schema and collection for SimpleSkill."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for skill evaluation."""

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy
        if self.precision is not None:
            result["precision"] = self.precision
        if self.recall is not None:
            result["recall"] = self.recall
        if self.f1 is not None:
            result["f1"] = self.f1
        result.update(self.custom)
        return result


@dataclass
class LatencyMetrics:
    """Latency metrics for skill execution."""

    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    samples: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "avg": self.avg_ms,
            "min": self.min_ms,
            "max": self.max_ms,
            "p50": self.p50_ms,
            "p90": self.p90_ms,
            "p95": self.p95_ms,
            "p99": self.p99_ms,
            "samples": self.samples,
        }


@dataclass
class CostMetrics:
    """Cost metrics for skill execution."""

    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    usd_estimate: float | None = None
    api_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            "api_calls": self.api_calls,
        }
        if self.usd_estimate is not None:
            result["usd_estimate"] = self.usd_estimate
        return result


@dataclass
class ReliabilityMetrics:
    """Reliability metrics for skill execution."""

    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    retry_count: int = 0
    total_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    @property
    def error_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.error_count / self.total_count

    @property
    def timeout_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.timeout_count / self.total_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_count": self.success_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "retry_count": self.retry_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
        }


@dataclass
class PrivacyMetrics:
    """Privacy-related metrics."""

    local_only: bool = True
    data_egress: bool = False
    pii_detected: bool = False
    encryption_used: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "local_only": self.local_only,
            "data_egress": self.data_egress,
            "pii_detected": self.pii_detected,
            "encryption_used": self.encryption_used,
        }


@dataclass
class ConformanceMetrics:
    """Conformance test results."""

    passed: bool = True
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "details": self.details,
        }


@dataclass
class SkillMetrics:
    """Complete metrics for a skill."""

    skill_name: str
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    reliability: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)
    privacy: PrivacyMetrics = field(default_factory=PrivacyMetrics)
    conformance: ConformanceMetrics = field(default_factory=ConformanceMetrics)
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "quality": self.quality.to_dict(),
            "latency": self.latency.to_dict(),
            "cost": self.cost.to_dict(),
            "reliability": self.reliability.to_dict(),
            "privacy": self.privacy.to_dict(),
            "conformance": self.conformance.to_dict(),
            "collected_at": self.collected_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillMetrics:
        metrics = cls(skill_name=data.get("skill_name", "unknown"))

        if "quality" in data:
            q = data["quality"]
            metrics.quality = QualityMetrics(
                accuracy=q.get("accuracy"),
                precision=q.get("precision"),
                recall=q.get("recall"),
                f1=q.get("f1"),
            )

        if "latency" in data:
            lat = data["latency"]
            metrics.latency = LatencyMetrics(
                avg_ms=lat.get("avg", 0),
                min_ms=lat.get("min", 0),
                max_ms=lat.get("max", 0),
                p50_ms=lat.get("p50", 0),
                p90_ms=lat.get("p90", 0),
                p95_ms=lat.get("p95", 0),
                p99_ms=lat.get("p99", 0),
                samples=lat.get("samples", 0),
            )

        if "reliability" in data:
            r = data["reliability"]
            metrics.reliability = ReliabilityMetrics(
                success_count=r.get("success_count", 0),
                error_count=r.get("error_count", 0),
                timeout_count=r.get("timeout_count", 0),
                retry_count=r.get("retry_count", 0),
                total_count=r.get("total_count", 0),
            )

        if "collected_at" in data:
            metrics.collected_at = datetime.fromisoformat(data["collected_at"])

        metrics.metadata = data.get("metadata", {})
        return metrics


class MetricsCollector:
    """Collects metrics during skill execution."""

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        self._latencies: list[float] = []
        self._metrics = SkillMetrics(skill_name=skill_name)
        self._current_start: float | None = None

    def measure_execution(self) -> _ExecutionContext:
        return _ExecutionContext(self)

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        self._update_latency_stats()

    def record_success(self) -> None:
        self._metrics.reliability.success_count += 1
        self._metrics.reliability.total_count += 1

    def record_error(self) -> None:
        self._metrics.reliability.error_count += 1
        self._metrics.reliability.total_count += 1

    def record_timeout(self) -> None:
        self._metrics.reliability.timeout_count += 1
        self._metrics.reliability.total_count += 1

    def record_retry(self) -> None:
        self._metrics.reliability.retry_count += 1

    def record_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._metrics.cost.tokens_input += input_tokens
        self._metrics.cost.tokens_output += output_tokens
        self._metrics.cost.tokens_total += input_tokens + output_tokens

    def record_api_call(self) -> None:
        self._metrics.cost.api_calls += 1

    def set_quality(
        self,
        accuracy: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        f1: float | None = None,
        **custom: float,
    ) -> None:
        if accuracy is not None:
            self._metrics.quality.accuracy = accuracy
        if precision is not None:
            self._metrics.quality.precision = precision
        if recall is not None:
            self._metrics.quality.recall = recall
        if f1 is not None:
            self._metrics.quality.f1 = f1
        self._metrics.quality.custom.update(custom)

    def set_privacy(
        self,
        local_only: bool | None = None,
        data_egress: bool | None = None,
        pii_detected: bool | None = None,
        encryption_used: bool | None = None,
    ) -> None:
        if local_only is not None:
            self._metrics.privacy.local_only = local_only
        if data_egress is not None:
            self._metrics.privacy.data_egress = data_egress
        if pii_detected is not None:
            self._metrics.privacy.pii_detected = pii_detected
        if encryption_used is not None:
            self._metrics.privacy.encryption_used = encryption_used

    def get_metrics(self) -> SkillMetrics:
        self._metrics.collected_at = datetime.now(UTC)
        return self._metrics

    def reset(self) -> None:
        self._latencies = []
        self._metrics = SkillMetrics(skill_name=self.skill_name)

    def _update_latency_stats(self) -> None:
        if not self._latencies:
            return

        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)

        self._metrics.latency.samples = n
        self._metrics.latency.avg_ms = sum(sorted_latencies) / n
        self._metrics.latency.min_ms = sorted_latencies[0]
        self._metrics.latency.max_ms = sorted_latencies[-1]
        self._metrics.latency.p50_ms = self._percentile(sorted_latencies, 50)
        self._metrics.latency.p90_ms = self._percentile(sorted_latencies, 90)
        self._metrics.latency.p95_ms = self._percentile(sorted_latencies, 95)
        self._metrics.latency.p99_ms = self._percentile(sorted_latencies, 99)

    @staticmethod
    def _percentile(sorted_values: list[float], p: int) -> float:
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


class _ExecutionContext:
    """Context manager for measuring execution time."""

    def __init__(self, collector: MetricsCollector) -> None:
        self._collector = collector
        self._start: float = 0.0

    def __enter__(self) -> _ExecutionContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._collector.record_latency(elapsed_ms)


class MetricsExporter:
    """Exports metrics to various formats."""

    @staticmethod
    def to_json(metrics: SkillMetrics, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.debug("Exported metrics to %s", path)

    @staticmethod
    def to_json_lines(
        metrics_list: list[SkillMetrics],
        path: Path | str,
        append: bool = False,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode) as f:
            for metrics in metrics_list:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        logger.debug("Exported %d metrics entries to %s", len(metrics_list), path)

    @staticmethod
    def to_opentelemetry_attributes(metrics: SkillMetrics) -> dict[str, Any]:
        attrs = {
            "skill.name": metrics.skill_name,
            "skill.latency.avg_ms": metrics.latency.avg_ms,
            "skill.latency.p95_ms": metrics.latency.p95_ms,
            "skill.reliability.success_rate": metrics.reliability.success_rate,
            "skill.reliability.error_rate": metrics.reliability.error_rate,
            "skill.cost.tokens_total": metrics.cost.tokens_total,
            "skill.cost.api_calls": metrics.cost.api_calls,
        }

        if metrics.quality.accuracy is not None:
            attrs["skill.quality.accuracy"] = metrics.quality.accuracy
        if metrics.quality.f1 is not None:
            attrs["skill.quality.f1"] = metrics.quality.f1

        return attrs


class MetricsStore:
    """In-memory store for skill metrics with aggregation."""

    def __init__(self) -> None:
        self._metrics: dict[str, list[SkillMetrics]] = {}

    def store(self, metrics: SkillMetrics) -> None:
        if metrics.skill_name not in self._metrics:
            self._metrics[metrics.skill_name] = []
        self._metrics[metrics.skill_name].append(metrics)

    def get_latest(self, skill_name: str) -> SkillMetrics | None:
        entries = self._metrics.get(skill_name, [])
        return entries[-1] if entries else None

    def get_all(self, skill_name: str) -> list[SkillMetrics]:
        return list(self._metrics.get(skill_name, []))

    def list_skills(self) -> list[str]:
        return list(self._metrics.keys())

    def aggregate(self, skill_name: str) -> SkillMetrics | None:
        entries = self._metrics.get(skill_name, [])
        if not entries:
            return None

        aggregated = SkillMetrics(skill_name=skill_name)

        all_latencies = []
        for m in entries:
            if m.latency.samples > 0:
                all_latencies.extend([m.latency.avg_ms] * m.latency.samples)

        if all_latencies:
            collector = MetricsCollector(skill_name)
            for lat in all_latencies:
                collector.record_latency(lat)
            aggregated.latency = collector.get_metrics().latency

        for m in entries:
            aggregated.reliability.success_count += m.reliability.success_count
            aggregated.reliability.error_count += m.reliability.error_count
            aggregated.reliability.timeout_count += m.reliability.timeout_count
            aggregated.reliability.retry_count += m.reliability.retry_count
            aggregated.reliability.total_count += m.reliability.total_count

        for m in entries:
            aggregated.cost.tokens_input += m.cost.tokens_input
            aggregated.cost.tokens_output += m.cost.tokens_output
            aggregated.cost.tokens_total += m.cost.tokens_total
            aggregated.cost.api_calls += m.cost.api_calls

        if entries[-1].quality:
            aggregated.quality = entries[-1].quality

        aggregated.metadata["aggregated_from"] = len(entries)
        aggregated.metadata["period_start"] = entries[0].collected_at.isoformat()
        aggregated.metadata["period_end"] = entries[-1].collected_at.isoformat()

        return aggregated

    def export_all(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for skill_name, entries in self._metrics.items():
            data[skill_name] = [m.to_dict() for m in entries]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug("Exported metrics for %d skills to %s", len(data), path)


__all__ = [
    "ConformanceMetrics",
    "CostMetrics",
    "LatencyMetrics",
    "MetricsCollector",
    "MetricsExporter",
    "MetricsStore",
    "PrivacyMetrics",
    "QualityMetrics",
    "ReliabilityMetrics",
    "SkillMetrics",
]
