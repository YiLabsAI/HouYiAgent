from houyi.application.context.session_context import SessionContextStateManager
from houyi.application.context.session_context_resolver import SessionContextResolver
from houyi.application.context.types import SessionContextState


class TestSessionContextResolver:
    def test_resolver_estimates_units(self):
        resolver = SessionContextResolver(
            state_manager=SessionContextStateManager(rolling_capacity=1000)
        )

        units = resolver.estimate_units(
            {"role": "user", "content": "hello world"},
            model="gpt-4o-mini",
        )

        assert units > 0

    def test_resolver_recovers_state(self):
        state_manager = SessionContextStateManager(rolling_capacity=1000)
        resolver = SessionContextResolver(state_manager=state_manager)

        state = resolver.recover_state(
            session_id="session-1",
            message_payloads=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            model="gpt-4o-mini",
            previous_state=SessionContextState(
                session_id="session-1",
                used_units=100,
                max_units=1000,
                state="healthy",
                last_compacted_at=8.0,
                last_compacted_message_count=4,
            ),
            updated_at=20.0,
        )

        assert state.session_id == "session-1"
        assert state.used_units > 0
        assert state.max_units == 1000
        assert state.last_compacted_at == 8.0
        assert state.last_compacted_message_count == 4
        assert state.updated_at == 20.0
