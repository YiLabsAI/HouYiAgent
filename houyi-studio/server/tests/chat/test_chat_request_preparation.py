from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from houyi_studio.server.chat.chat_request_preparation import ChatRequestPreparation
from houyi_studio.server.chat.types import MessageRole, SendMessageRequest


class _Lock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Span:
    def __init__(self, *, trace_id: str = "trace-1"):
        self.trace_id = trace_id
        self.attributes: dict[str, object] = {}
        self.status: tuple[str, str | None] | None = None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: str, message: str | None = None) -> None:
        self.status = (status, message)


class TestChatRequestPreparation:
    @pytest.mark.asyncio
    async def test_prepare_builds_context(self):
        lock = _Lock()
        conversation = SimpleNamespace(
            conversation_id="conv-1",
            model="conv-model",
            system_instructions="conv-sys",
            messages=[],
            updated_at=0.0,
        )
        conversation_snapshot = SimpleNamespace(messages=["snap"])
        conversation.model_copy = MagicMock(return_value=conversation_snapshot)
        compacted_snapshot = SimpleNamespace(messages=["compacted"])
        runtime_profile = SimpleNamespace(
            name="balanced",
            keep_n=8,
            low_watermark=200,
            compression_threshold=0.7,
            overflow_threshold=0.9,
            cooldown_messages=2,
            cooldown_seconds=30,
            tool_result_max_tokens=256,
        )
        json_store = SimpleNamespace(
            lock=AsyncMock(return_value=lock),
            get=MagicMock(return_value=conversation),
            update=MagicMock(),
        )
        deps = SimpleNamespace(
            default_model="default-model",
            default_system_instructions="default-sys",
            conversation_context=SimpleNamespace(
                estimate_units=MagicMock(return_value=9),
            ),
            context_state_updater=SimpleNamespace(
                apply=MagicMock(
                    return_value=SimpleNamespace(event_payload={"conversation_id": "conv-1"})
                ),
                request_cls=MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            ),
            resolve_llm_kwargs=MagicMock(
                return_value=({"temperature": 0.3}, {"input_budget": 321})
            ),
            resolve_runtime_profile=MagicMock(return_value=runtime_profile),
            context_compressor=SimpleNamespace(
                compact_for_send=AsyncMock(
                    return_value=SimpleNamespace(
                        conversation_snapshot=compacted_snapshot,
                        compaction_event={"kind": "compacted"},
                        context_state_event={
                            "conversation_id": "conv-1",
                            "source": "release_delta",
                        },
                    )
                )
            ),
            build_context_messages=MagicMock(
                return_value=([{"role": "system", "content": "sys"}], {"used_tokens": 12})
            ),
        )
        preparation = ChatRequestPreparation(
            json_store=json_store,
            default_model=deps.default_model,
            default_system_instructions=deps.default_system_instructions,
            conversation_context=deps.conversation_context,
            context_state_updater=deps.context_state_updater,
            resolve_llm_kwargs=deps.resolve_llm_kwargs,
            resolve_runtime_profile=deps.resolve_runtime_profile,
            context_compressor=deps.context_compressor,
            build_context_messages=deps.build_context_messages,
        )

        prepared = await preparation.prepare(
            conversation_id="conv-1",
            request=SendMessageRequest(content="hello", model="req-model"),
            chat_span=_Span(),
        )

        assert prepared.model == "req-model"
        assert prepared.llm_kwargs == {"temperature": 0.3}
        assert prepared.context_usage == {"used_tokens": 12}
        assert prepared.context_state_event == {"conversation_id": "conv-1"}
        assert prepared.compaction_event == {"kind": "compacted"}
        assert prepared.compaction_state_event == {
            "conversation_id": "conv-1",
            "source": "release_delta",
        }
        assert conversation.messages[-1].role == MessageRole.USER
        assert conversation.messages[-1].metadata["usage"]["total_tokens"] == 9
        deps.context_state_updater.request_cls.assert_called_once_with(
            mode="append",
            reason="user_append",
            model="req-model",
            messages=[conversation.messages[-1]],
        )
        deps.context_state_updater.apply.assert_called_once()
        deps.build_context_messages.assert_called_once_with(
            conversation=compacted_snapshot,
            model="req-model",
            sys_instructions="conv-sys",
            span=ANY,
            input_budget=321,
        )
