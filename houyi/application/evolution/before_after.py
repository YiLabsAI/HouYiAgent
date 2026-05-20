"""Before/after report for an evolution optimization run."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from houyi.application.evolution.artifacts import EvolutionArtifact


@dataclass(frozen=True, slots=True)
class BeforeAfterReport:
    run_id: str
    optimizer: str
    artifact_type: str
    baseline_content: str
    optimized_content: str
    baseline_score: float
    optimized_score: float
    delta: float
    sample_size: int
    signal_count: int
    verdict: str
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metrics"] = dict(self.metrics)
        return payload

    def render_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Evolution before/after — {self.run_id}")
        lines.append("")
        lines.append(f"- optimizer: `{self.optimizer}`")
        lines.append(f"- artifact_type: `{self.artifact_type}`")
        lines.append(f"- verdict: **{self.verdict}** ({self.reason})")
        lines.append(f"- baseline score: {self.baseline_score:.4f}")
        lines.append(f"- optimized score: {self.optimized_score:.4f}")
        lines.append(f"- delta: {self.delta:+.4f}")
        lines.append(f"- sample_size: {self.sample_size}")
        lines.append(f"- signal_count: {self.signal_count}")
        if self.metrics:
            lines.append("")
            lines.append("## metrics")
            for key, value in sorted(self.metrics.items()):
                lines.append(f"- {key}: {value:.4f}")
        lines.append("")
        lines.append("## baseline content")
        lines.append("")
        lines.append("```")
        lines.append(self.baseline_content)
        lines.append("```")
        lines.append("")
        lines.append("## optimized content")
        lines.append("")
        lines.append("```")
        lines.append(self.optimized_content)
        lines.append("```")
        lines.append("")
        return "\n".join(lines)


def write_report(report: BeforeAfterReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "before_after.md"
    json_path = output_dir / "before_after.json"
    md_path.write_text(report.render_markdown(), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return md_path


def make_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def baseline_artifact_summary(artifact: EvolutionArtifact) -> str:
    return f"{artifact.artifact_type.value}#{artifact.artifact_id[:12]} v{artifact.version}"


__all__ = [
    "BeforeAfterReport",
    "baseline_artifact_summary",
    "make_run_id",
    "write_report",
]
