"""Chat data types for Studio Server layer.

These are the server-side persistence and API models. They wrap SDK types
where needed but own the storage schema (JSON Store format).
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# MIME types / extensions whose base64 payload is decodable as UTF-8 text.
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXACT = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/typescript",
        "application/x-python",
        "application/x-sh",
        "application/x-yaml",
        "application/toml",
        "application/sql",
    }
)
# Extensions that are text-extractable even if MIME is generic
_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".py",
        ".rb",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".swift",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".sql",
        ".graphql",
        ".proto",
        ".r",
        ".lua",
        ".pl",
        ".pm",
        ".tex",
        ".bib",
        ".rst",
        ".adoc",
        ".org",
        ".log",
        ".diff",
        ".patch",
    }
)


class MessageRole(str, Enum):
    """Role of a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Attachment(BaseModel):
    """A file attachment on a message (image, document, etc.).

    Images are stored as base64 data URIs for portability.
    """

    filename: str
    mime_type: str
    data: str  # base64-encoded data URI (e.g. "data:image/png;base64,...")
    size: int = 0  # original file size in bytes


class Message(BaseModel):
    """A single chat message in a conversation."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: MessageRole
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    bookmarked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @staticmethod
    def _is_text_extractable(att: Attachment) -> bool:
        """Return True if the attachment content can be decoded as UTF-8 text."""
        if any(att.mime_type.startswith(p) for p in _TEXT_MIME_PREFIXES):
            return True
        if att.mime_type in _TEXT_MIME_EXACT:
            return True
        dot = att.filename.rfind(".")
        return dot >= 0 and att.filename[dot:].lower() in _TEXT_EXTENSIONS

    @staticmethod
    def _extract_text(att: Attachment) -> str | None:
        """Decode a base64 data-URI attachment and return its text content.

        Returns None if decoding fails (binary file, corrupt data, etc.).
        """
        try:
            data = att.data
            # Strip data URI prefix: "data:<mime>;base64,<payload>"
            if data.startswith("data:"):
                _, _, payload = data.partition(";base64,")
                if not payload:
                    return None
            else:
                payload = data
            raw = base64.b64decode(payload)
            return raw.decode("utf-8")
        except Exception:
            logger.debug("Cannot extract text from %s", att.filename)
            return None

    def to_llm_message(self, *, vision: bool = True) -> dict[str, Any]:
        """Convert to LLM-compatible message dict.

        Attachment handling strategy:

        * **Images** – vision models receive ``image_url`` parts; non-vision
          models get a ``[Image: filename]`` text placeholder.
        * **Text files** (source code, markdown, CSV, JSON, …) – base64
          payload is decoded to UTF-8 and injected as a text block so the
          model can read the file content.
        * **Binary documents** (PDF, Word, Excel, …) – only a filename
          description is sent because we have no server-side parser yet.
          A future version may add PDF text extraction.

        Args:
            vision: If True, image attachments are sent as OpenAI multimodal
                ``image_url`` parts.  If False (model does not support vision),
                images are described as text placeholders.
        """

        def _attach_tool_fields(payload: dict[str, Any]) -> dict[str, Any]:
            if self.role == MessageRole.ASSISTANT and isinstance(self.reasoning_content, str):
                payload["reasoning_content"] = self.reasoning_content
            if self.role == MessageRole.ASSISTANT and self.tool_calls:
                payload["tool_calls"] = self.tool_calls
            if self.role == MessageRole.TOOL:
                if self.tool_call_id:
                    payload["tool_call_id"] = self.tool_call_id
                if self.name:
                    payload["name"] = self.name
            return payload

        if not self.attachments:
            return _attach_tool_fields({"role": self.role.value, "content": self.content})

        # Collect text snippets for non-multimodal parts and multimodal parts
        text_snippets: list[str] = []
        multimodal_parts: list[dict[str, Any]] = []

        for att in self.attachments:
            # --- Images ---
            if att.mime_type.startswith("image/"):
                if vision:
                    multimodal_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": att.data},
                        }
                    )
                else:
                    text_snippets.append(f"[Image: {att.filename}]")
                continue

            # --- Text-extractable files ---
            if self._is_text_extractable(att):
                extracted = self._extract_text(att)
                if extracted is not None:
                    # Truncate very large files to avoid blowing up context
                    max_chars = 100_000
                    if len(extracted) > max_chars:
                        extracted = extracted[:max_chars] + "\n... (truncated)"
                    text_snippets.append(f"--- file: {att.filename} ---\n{extracted}\n--- end ---")
                    continue

            # --- Binary documents (PDF, Word, Excel, etc.) ---
            text_snippets.append(
                f"[Attached file: {att.filename} ({att.mime_type}, "
                f"{att.size} bytes) — content not extractable]"
            )

        # Merge everything into the final message
        extra_text = "\n\n".join(text_snippets) if text_snippets else ""
        full_text = (
            (self.content + "\n\n" + extra_text)
            if self.content and extra_text
            else self.content or extra_text
        )

        if multimodal_parts:
            # Vision mode with image parts — use content array
            if full_text:
                multimodal_parts.append({"type": "text", "text": full_text})
            return _attach_tool_fields(
                {
                    "role": self.role.value,
                    "content": multimodal_parts if multimodal_parts else full_text,
                }
            )

        # No multimodal parts — plain text message
        return _attach_tool_fields({"role": self.role.value, "content": full_text})


class ConversationStatus(str, Enum):
    """Status of a conversation."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class Conversation(BaseModel):
    """A chat conversation with message history.

    Persisted as a single JSON file per conversation.
    """

    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New Chat"
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: list[Message] = Field(default_factory=list)
    model: str = ""
    system_instructions: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool | None = None
    bookmarked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # Schema version for forward compatibility
    schema_version: int = 1

    @property
    def message_count(self) -> int:
        """Number of messages in this conversation."""
        return len(self.messages)

    @property
    def visible_message_count(self) -> int:
        """Message count aligned with the chat timeline's primary visible items."""
        return sum(
            1
            for message in self.messages
            if message.role not in {MessageRole.SYSTEM, MessageRole.TOOL}
        )

    @property
    def last_message_at(self) -> float | None:
        """Timestamp of the last message, or None if empty."""
        if not self.messages:
            return None
        return self.messages[-1].created_at

    def to_summary(self) -> dict[str, Any]:
        """Return a lightweight summary (for list endpoints)."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "status": self.status.value,
            "message_count": self.message_count,
            "visible_message_count": self.visible_message_count,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_message_at": self.last_message_at,
            "bookmarked": self.bookmarked,
        }

    def get_llm_messages(self) -> list[dict[str, str]]:
        """Convert all messages to LLM-compatible format."""
        return [m.to_llm_message() for m in self.messages]


# --- API request/response models ---


class CreateConversationRequest(BaseModel):
    """Request body for creating a new conversation."""

    title: str = "New Chat"
    model: str = ""
    system_instructions: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    """Request body for sending a message."""

    content: str
    attachments: list[Attachment] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    enable_reasoning: bool | None = None
    enable_tool_calls: bool | None = None
    tool_call_strategy: str | None = Field(
        default=None,
        pattern=r"^(conservative|balanced|aggressive)$",
    )
    enable_web_search: bool | None = None
    enable_skills: list[str] | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=50)


class UpdateConversationRequest(BaseModel):
    """Request body for updating conversation metadata."""

    title: str | None = None
    status: ConversationStatus | None = None
    system_instructions: str | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool | None = None
    bookmarked: bool | None = None


class EditMessageRequest(BaseModel):
    """Request body for editing a message's content."""

    content: str
