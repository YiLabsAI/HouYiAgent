from houyi.infrastructure.config import (
    ENV_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS,
    ENV_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS,
    EnvConfig,
    env,
)
from houyi.infrastructure.config.env_config import (
    ENV_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS as LEGACY_MAX_MESSAGE_CHARS,
)
from houyi.infrastructure.config.env_config import (
    ENV_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS as LEGACY_MAX_TOTAL_CHARS,
)


def test_infrastructure_config_exports_canonical_symbols() -> None:
    assert EnvConfig is not None
    assert env.rag_knowledge_dir is not None
    assert ENV_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS == LEGACY_MAX_MESSAGE_CHARS
    assert ENV_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS == LEGACY_MAX_TOTAL_CHARS
