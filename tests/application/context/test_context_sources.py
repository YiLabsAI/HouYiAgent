from houyi.application.context.context_sources import (
    assemble_context_candidates,
    build_history_message_payloads,
    build_pinned_context_candidates,
    build_tool_summary_candidates,
    extract_latest_compaction_summary,
    is_summarized_tool_message,
)
from houyi.application.context.session_context import SessionContextStateManager
from houyi.application.context.types import ContextBlockType, ContextCandidate, ContextSourceKind


class TestSessionContextStateManager:
    def test_state_init(self):
        manager = SessionContextStateManager(rolling_capacity=1000)

        state = manager.build_initial_state("session-1", now=12.0)

        assert state.session_id == "session-1"
        assert state.used_units == 0
        assert state.max_units == 1000
        assert state.state == "healthy"
        assert state.updated_at == 12.0

    def test_state_recover(self):
        manager = SessionContextStateManager(rolling_capacity=1000)

        state = manager.recover_state(
            session_id="session-1",
            used_units=720,
            previous_state=manager.build_initial_state("session-1", now=8.0),
            updated_at=20.0,
        )

        assert state.session_id == "session-1"
        assert state.used_units == 720
        assert state.max_units == 1000
        assert state.state == "elevated"
        assert state.updated_at == 20.0

    def test_state_normalize(self):
        manager = SessionContextStateManager(rolling_capacity=1000)

        state = manager.normalize_state(
            session_id="session-1",
            state=manager.build_initial_state("old-id", now=0.0).model_copy(
                update={
                    "used_units": 5000,
                    "max_units": 5,
                    "last_compaction_delta": 3,
                    "last_compacted_message_count": 7,
                }
            ),
            updated_at=9.0,
        )

        assert state.session_id == "session-1"
        assert state.used_units == 1000
        assert state.max_units == 1000
        assert state.state == "compacted_recently"
        assert state.last_compacted_message_count == 7
        assert state.updated_at == 9.0

    def test_state_delta(self):
        manager = SessionContextStateManager(rolling_capacity=1000)

        state = manager.apply_delta(
            state=manager.build_initial_state("session-1", now=1.0).model_copy(
                update={
                    "used_units": 920,
                    "state": "near_compaction",
                }
            ),
            released_units=400,
            compacted_at=22.0,
            compaction_delta=400,
            compacted_message_count=12,
            now=30.0,
        )

        assert state.used_units == 520
        assert state.last_compacted_at == 22.0
        assert state.last_compaction_delta == 400
        assert state.last_compacted_message_count == 12
        assert state.state == "compacted_recently"
        assert state.updated_at == 30.0


class TestContextSources:
    def test_summary_extracts_latest(self):
        summary, metadata = extract_latest_compaction_summary(
            {
                "compaction_history": [
                    {"summary": "", "trigger": "manual"},
                    {
                        "compaction_id": "cmp_2",
                        "backup_id": "bck_2",
                        "trigger": "pre_request_pressure",
                        "pressure_level": "high",
                        "summary": "Latest summary",
                        "metadata": {"summary_model": "gpt-4o-mini"},
                    },
                ]
            }
        )

        assert summary == "Latest summary"
        assert metadata["compaction_id"] == "cmp_2"
        assert metadata["backup_id"] == "bck_2"
        assert metadata["trigger"] == "pre_request_pressure"
        assert metadata["pressure_level"] == "high"
        assert metadata["summary_model"] == "gpt-4o-mini"

    def test_summary_skips_invalid(self):
        summary, metadata = extract_latest_compaction_summary(
            {"compaction_history": ["bad", {"summary": "  "}]}
        )

        assert summary is None
        assert metadata == {}

    def test_pin_builds_active(self):
        candidates = build_pinned_context_candidates(
            {
                "pinned_contexts": [
                    {
                        "pin_id": "pin_1",
                        "source_message_id": "u1",
                        "content": "Deploy to staging first.",
                        "status": "active",
                        "priority": 7,
                        "token_count": 12,
                        "metadata": {"origin_message_id": "u1"},
                    }
                ]
            }
        )

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.source == ContextSourceKind.PINNED
        assert candidate.block_type == ContextBlockType.PINNED
        assert candidate.content == "Deploy to staging first."
        assert candidate.pinned is True
        assert candidate.token_count == 12
        assert candidate.metadata["pin_id"] == "pin_1"
        assert candidate.metadata["source_message_id"] == "u1"
        assert candidate.metadata["pin_priority"] == 7
        assert candidate.metadata["origin_message_id"] == "u1"

    def test_pin_skips_inactive(self):
        candidates = build_pinned_context_candidates(
            {
                "pinned_contexts": [
                    {"content": "Active", "status": "active"},
                    {"content": "Archived", "status": "archived"},
                    {"content": "", "status": "active"},
                ]
            }
        )

        assert len(candidates) == 1
        assert candidates[0].content == "Active"

    def test_candidates_build_structured(self):
        candidates = assemble_context_candidates(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ],
            system_instructions="You are helpful",
            memory_context="User prefers concise replies",
            boundary_id="boundary-chat",
        )

        assert [candidate.source for candidate in candidates] == [
            ContextSourceKind.SYSTEM,
            ContextSourceKind.MEMORY,
            ContextSourceKind.CURRENT_TURN,
            ContextSourceKind.RECENT,
            ContextSourceKind.RECENT,
        ]
        assert [candidate.block_type for candidate in candidates] == [
            ContextBlockType.SYSTEM,
            ContextBlockType.MEMORY,
            ContextBlockType.RECENT,
            ContextBlockType.RECENT,
            ContextBlockType.RECENT,
        ]
        assert candidates[2].content == [{"role": "user", "content": "How are you?"}]
        assert candidates[3].content == [{"role": "user", "content": "Hello"}]
        assert candidates[4].content == [{"role": "assistant", "content": "Hi there"}]
        assert candidates[2].metadata["boundary_id"] == "boundary-chat"
        assert candidates[3].metadata["boundary_id"] == "boundary-chat"
        assert candidates[4].metadata["boundary_id"] == "boundary-chat"

    def test_candidates_build_summary(self):
        candidates = assemble_context_candidates(
            messages=[{"role": "user", "content": "How are you?"}],
            system_instructions="You are helpful",
            memory_context=None,
            summary_context="Earlier decisions",
            summary_metadata={"compaction_id": "cmp_1"},
        )

        assert [candidate.source for candidate in candidates] == [
            ContextSourceKind.SYSTEM,
            ContextSourceKind.SUMMARY,
            ContextSourceKind.CURRENT_TURN,
        ]
        assert candidates[1].block_type == ContextBlockType.SUMMARY
        assert candidates[1].content == "Earlier decisions"
        assert candidates[1].metadata["compaction_id"] == "cmp_1"

    def test_tool_summary_builds(self):
        messages = [
            {
                "message_id": "m1",
                "role": "tool",
                "name": "search",
                "tool_call_id": "call-1",
                "metadata": {
                    "tool_result_profile": {
                        "compressed": True,
                        "summary": "2 results summarized",
                    }
                },
            }
        ]

        candidates, summarized_ids = build_tool_summary_candidates(
            messages,
            boundary_id="boundary-chat",
        )

        assert summarized_ids == {"call-1"}
        assert len(candidates) == 1
        candidate = candidates[0]
        assert isinstance(candidate, ContextCandidate)
        assert candidate.source == ContextSourceKind.TOOL_SUMMARY
        assert candidate.block_type == ContextBlockType.TOOL_SUMMARY
        assert candidate.content == "search: 2 results summarized"
        assert candidate.metadata["boundary_id"] == "boundary-chat"
        assert candidate.metadata["tool_call_id"] == "call-1"

    def test_tool_summary_detects(self):
        message = {
            "message_id": "m1",
            "role": "tool",
            "tool_call_id": "call-1",
        }

        assert is_summarized_tool_message(message, {"call-1"}) is True
        assert is_summarized_tool_message(message, {"m1"}) is True
        assert is_summarized_tool_message(message, {"call-2"}) is False

    def test_history_skips_summarized_tool(self):
        messages = [
            {"role": "user", "content": "Find config"},
            {
                "message_id": "tool-1",
                "role": "tool",
                "name": "read_file",
                "tool_call_id": "call-1",
                "content": "RAW TOOL PAYLOAD",
                "metadata": {
                    "tool_result_profile": {
                        "compressed": True,
                        "summary": "Found config path",
                    }
                },
            },
            {"role": "assistant", "content": "I found it"},
        ]

        payloads = build_history_message_payloads(
            messages,
            message_to_payload=lambda message: dict(message),
        )

        assert [payload["role"] for payload in payloads] == ["user", "assistant"]
        assert all(payload.get("content") != "RAW TOOL PAYLOAD" for payload in payloads)
