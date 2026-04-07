from __future__ import annotations

import json
from dataclasses import dataclass, field

from houyi.application.research.runtime.processing import (
    process_agent_search_output,
    process_sub_question_execution_order,
)


@dataclass
class _Question:
    question_id: str
    priority: float
    expected_sources: int = 2
    depends_on: list[str] = field(default_factory=list)


class TestExecutionOrder:
    def test_orders_dependencies(self):
        questions = [
            _Question("q2", priority=0.7, depends_on=["q1"]),
            _Question("q1", priority=0.2),
            _Question("q3", priority=0.9),
        ]

        ordered = process_sub_question_execution_order(questions)

        assert [q.question_id for q in ordered] == ["q3", "q1", "q2"]

    def test_ignores_missing_dependency(self):
        questions = [
            _Question("q2", priority=0.7, depends_on=["q_missing"]),
            _Question("q1", priority=0.2),
        ]

        ordered = process_sub_question_execution_order(questions)

        assert [q.question_id for q in ordered] == ["q2", "q1"]


class TestSearchOutput:
    def test_parses_json_output(self):
        question = _Question("q1", priority=1.0, expected_sources=2)
        output = json.dumps(
            {
                "sources": [
                    {
                        "url": "https://a.com",
                        "title": "A",
                        "snippet": "Alpha",
                    }
                ],
                "summary": "Summary text",
                "queries_used": ["query one"],
            }
        )

        result = process_agent_search_output(question, output)

        assert result.question_id == "q1"
        assert len(result.sources) == 1
        assert result.sources[0].snippet == "Alpha"
        assert result.summary == "Summary text"
        assert result.rounds[0].queries == ["query one"]
        assert result.coverage_score == 0.5
        assert result.exhausted is False

    def test_parses_fenced_json(self):
        question = _Question("q2", priority=1.0, expected_sources=1)
        output = """```json
{
  \"sources\": [
    {
      \"url\": \"https://b.com\",
      \"title\": \"B\",
      \"content_summary\": \"Beta\"
    }
  ],
  \"summary\": \"Fenced summary\",
  \"queries_used\": [\"query two\"]
}
```"""

        result = process_agent_search_output(question, output)

        assert result.sources[0].snippet == "Beta"
        assert result.summary == "Fenced summary"
        assert result.rounds[0].queries == ["query two"]
        assert result.coverage_score == 1.0

    def test_falls_back_on_invalid_json(self):
        question = _Question("q3", priority=1.0, expected_sources=3)
        output = "x" * 700

        result = process_agent_search_output(question, output)

        assert result.sources == []
        assert result.summary == output[:500]
        assert result.rounds[0].rationale == output[:200]
        assert result.coverage_score == 0.0
        assert result.exhausted is True
