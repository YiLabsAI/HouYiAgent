"""Import/Export for chat conversations.

Supports importing from compatible backup formats (zip with data.json)
and exporting HouYi conversations as JSON.

Invariants:
- Order preserved: messages within a conversation maintain original timestamp order
- Content preserved: non-text blocks become placeholders + raw_import metadata
- Rollback: imported conversations can be bulk-deleted
"""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from datetime import UTC
from typing import Any

from .json_store import JsonStore
from .types import Conversation, ConversationStatus, Message, MessageRole

logger = logging.getLogger(__name__)


class CherryStudioImporter:
    """Imports CherryStudio backup zip into HouYi conversation store.

    CherryStudio backup structure:
    - ZIP containing data.json at top level
    - data.json has indexedDB.topics (conversations) and
      indexedDB.message_blocks (normalized message blocks)

    Cherry → HouYi mapping:
    - topic.id → conversation_id, topic.title → title
    - message_blocks grouped by message_id → Message
    - Block types: text → content, think → reasoning_content,
      others → placeholder + raw_import
    """

    def __init__(self, json_store: JsonStore):
        self.json_store = json_store

    def import_from_zip(self, zip_data: bytes) -> ImportResult:
        """Import conversations from a CherryStudio backup zip.

        Args:
            zip_data: Raw bytes of the zip file.

        Returns:
            ImportResult with counts and any warnings.
        """
        result = ImportResult()

        try:
            data = self._extract_data_json(zip_data)
        except Exception as e:
            result.errors.append(f"Failed to extract data.json: {e}")
            return result

        indexed_db = data.get("indexedDB", {})
        topics = indexed_db.get("topics", [])
        message_blocks = indexed_db.get("message_blocks", [])

        if not topics:
            result.warnings.append("No topics found in backup")
            return result

        # Index message_blocks by message_id for fast lookup
        blocks_by_message: dict[str, list[dict]] = {}
        for block in message_blocks:
            msg_id = block.get("messageId") or block.get("message_id", "")
            if msg_id:
                blocks_by_message.setdefault(msg_id, []).append(block)

        # Sort blocks within each message by creation order
        for msg_id in blocks_by_message:
            blocks_by_message[msg_id].sort(key=lambda b: b.get("createdAt", b.get("created_at", 0)))

        for topic in topics:
            try:
                conversation = self._convert_topic(topic, blocks_by_message)
                self.json_store.create(conversation)
                result.conversations_imported += 1
                result.messages_imported += len(conversation.messages)
            except ValueError as e:
                # Conversation already exists
                result.warnings.append(f"Skipped topic {topic.get('id', '?')}: {e}")
            except Exception as e:
                result.errors.append(f"Failed to import topic {topic.get('id', '?')}: {e}")

        logger.info(
            "CherryStudio import: %d conversations, %d messages, %d warnings, %d errors",
            result.conversations_imported,
            result.messages_imported,
            len(result.warnings),
            len(result.errors),
        )
        return result

    def import_from_json(self, json_data: dict[str, Any]) -> ImportResult:
        """Import from already-parsed data.json content.

        Args:
            json_data: Parsed data.json dict.

        Returns:
            ImportResult with counts and any warnings.
        """
        result = ImportResult()
        indexed_db = json_data.get("indexedDB", {})
        topics = indexed_db.get("topics", [])
        message_blocks = indexed_db.get("message_blocks", [])

        if not topics:
            result.warnings.append("No topics found in data")
            return result

        blocks_by_message: dict[str, list[dict]] = {}
        for block in message_blocks:
            msg_id = block.get("messageId") or block.get("message_id", "")
            if msg_id:
                blocks_by_message.setdefault(msg_id, []).append(block)

        for msg_id in blocks_by_message:
            blocks_by_message[msg_id].sort(key=lambda b: b.get("createdAt", b.get("created_at", 0)))

        for topic in topics:
            try:
                conversation = self._convert_topic(topic, blocks_by_message)
                self.json_store.create(conversation)
                result.conversations_imported += 1
                result.messages_imported += len(conversation.messages)
            except ValueError as e:
                result.warnings.append(f"Skipped topic {topic.get('id', '?')}: {e}")
            except Exception as e:
                result.errors.append(f"Failed to import topic {topic.get('id', '?')}: {e}")

        return result

    @staticmethod
    def _extract_data_json(zip_data: bytes) -> dict[str, Any]:
        """Extract and parse data.json from zip."""
        with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
            # Look for data.json at top level
            for name in zf.namelist():
                if name.endswith("data.json") and "/" not in name.rstrip("/"):
                    raw = zf.read(name)
                    return json.loads(raw)
            raise FileNotFoundError("data.json not found in zip")

    def _convert_topic(
        self,
        topic: dict[str, Any],
        blocks_by_message: dict[str, list[dict]],
    ) -> Conversation:
        """Convert a CherryStudio topic to a HouYi Conversation.

        Args:
            topic: CherryStudio topic dict.
            blocks_by_message: Message blocks indexed by message_id.

        Returns:
            HouYi Conversation.
        """
        topic_id = str(topic.get("id", ""))
        title = topic.get("title", topic.get("name", "Imported Chat"))

        # Extract messages from topic
        # CherryStudio topics have a 'messages' array with message references
        raw_messages = topic.get("messages", [])

        messages: list[Message] = []
        for raw_msg in raw_messages:
            msg = self._convert_message(raw_msg, blocks_by_message)
            if msg is not None:
                messages.append(msg)

        # Sort by created_at to maintain order invariant
        messages.sort(key=lambda m: m.created_at)

        # Timestamps
        created_ts = topic.get("createdAt", topic.get("created_at", 0))
        updated_ts = topic.get("updatedAt", topic.get("updated_at", 0))
        if isinstance(created_ts, (int, float)) and created_ts > 1e12:
            created_ts = created_ts / 1000  # ms → s
        if isinstance(updated_ts, (int, float)) and updated_ts > 1e12:
            updated_ts = updated_ts / 1000

        now = time.time()
        return Conversation(
            conversation_id=topic_id or None,  # type: ignore[arg-type]
            title=str(title)[:200],
            status=ConversationStatus.ACTIVE,
            messages=messages,
            model=topic.get("model", ""),
            metadata={
                "import_source": "cherrystudio",
                "import_time": now,
                "raw_import": {k: v for k, v in topic.items() if k not in ("messages",)},
            },
            created_at=float(created_ts) if created_ts else now,
            updated_at=float(updated_ts) if updated_ts else now,
        )

    def _convert_message(
        self,
        raw_msg: dict[str, Any],
        blocks_by_message: dict[str, list[dict]],
    ) -> Message | None:
        """Convert a CherryStudio message to a HouYi Message.

        Args:
            raw_msg: CherryStudio message dict.
            blocks_by_message: Block lookup.

        Returns:
            HouYi Message, or None if message should be skipped.
        """
        msg_id = str(raw_msg.get("id", ""))
        role = self._infer_role(raw_msg, blocks_by_message.get(msg_id, []))

        # Assemble content from blocks
        blocks = blocks_by_message.get(msg_id, [])
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        raw_blocks: list[dict] = []

        if blocks:
            for block in blocks:
                block_type = block.get("type", "text")
                block_content = block.get("content", "")

                if block_type == "text":
                    content_parts.append(str(block_content))
                elif block_type == "think":
                    reasoning_parts.append(str(block_content))
                elif block_type == "image":
                    file_id = block.get("fileId", block.get("file_id", ""))
                    content_parts.append(f"[image: {file_id}]")
                elif block_type == "code":
                    lang = block.get("language", "")
                    content_parts.append(f"```{lang}\n{block_content}\n```")
                elif block_type in ("tooluse", "tool_use"):
                    tool_name = block.get("toolName", block.get("tool_name", "tool"))
                    content_parts.append(f"[tool_use: {tool_name}]")
                elif block_type in ("toolresult", "tool_result"):
                    content_parts.append(f"[tool_result: {block_content[:200]}]")
                else:
                    content_parts.append(f"[{block_type}: {str(block_content)[:100]}]")

                raw_blocks.append(block)
        else:
            # Fallback: use message-level content
            content = raw_msg.get("content", "")
            if content:
                content_parts.append(str(content))

        if not content_parts and not reasoning_parts:
            return None

        # Timestamp
        created_ts = raw_msg.get("createdAt", raw_msg.get("created_at", 0))
        if isinstance(created_ts, (int, float)) and created_ts > 1e12:
            created_ts = created_ts / 1000

        metadata: dict[str, Any] = {}
        if raw_blocks:
            metadata["raw_import"] = {"blocks": raw_blocks}

        return Message(
            message_id=msg_id or None,  # type: ignore[arg-type]
            role=role,
            content="\n".join(content_parts),
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            metadata=metadata,
            created_at=float(created_ts) if created_ts else time.time(),
        )

    @staticmethod
    def _infer_role(
        raw_msg: dict[str, Any],
        blocks: list[dict],
    ) -> MessageRole:
        """Infer message role from CherryStudio message data.

        Priority:
        1. Explicit role field
        2. Block types (tool_use/tool_result → tool)
        3. Author field
        4. Default to user
        """
        # Check explicit role
        role_str = raw_msg.get("role", "").lower()
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.TOOL,
        }
        if role_str in role_map:
            return role_map[role_str]

        # Check blocks for tool indicators
        for block in blocks:
            bt = block.get("type", "")
            if bt in ("tooluse", "tool_use", "toolresult", "tool_result"):
                return MessageRole.TOOL

        # Check author field
        author = raw_msg.get("author", "").lower()
        if author in ("user", "human"):
            return MessageRole.USER
        if author in ("assistant", "ai", "bot"):
            return MessageRole.ASSISTANT

        return MessageRole.USER


class ChatExporter:
    """Exports all conversations as HouyiChatWorkspace JSON.

    Output format uses the HouYiChatWorkspace schema:
    {
      "version": 1,
      "exported_at": ISO timestamp,
      "conversations": [...],
      "settings": {},
      "assistants": [],
      "memories": [],
    }

    Thread-safe: reads only, no mutations.
    """

    SCHEMA_VERSION = 1

    def __init__(self, json_store: JsonStore):
        self._store = json_store

    def export_all(self) -> dict[str, Any]:
        """Export all conversations as a HouyiChatWorkspace dict.

        Returns:
            Dict matching HouyiChatWorkspace schema, JSON-serializable.
        """
        from datetime import datetime

        conversations = self._store.list_conversations(limit=10000, offset=0)
        full_conversations = []
        for summary in conversations:
            conv_id = (
                summary.get("conversation_id")
                if isinstance(summary, dict)
                else getattr(summary, "conversation_id", None)
            )
            if conv_id:
                conv = self._store.get(conv_id)
                if conv:
                    full_conversations.append(conv.model_dump(mode="json"))

        return {
            "version": self.SCHEMA_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "conversations": full_conversations,
            "assistants": [],
            "settings": {},
            "memories": [],
        }

    def export_json_bytes(self) -> bytes:
        """Export as formatted JSON bytes (UTF-8).

        Returns:
            JSON bytes suitable for file download.
        """
        data = self.export_all()
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


class ImportResult:
    """Result of a CherryStudio import operation."""

    def __init__(self):
        self.conversations_imported: int = 0
        self.messages_imported: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "conversations_imported": self.conversations_imported,
            "messages_imported": self.messages_imported,
            "warnings": self.warnings,
            "errors": self.errors,
        }
