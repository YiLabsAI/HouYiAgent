"""Unit tests for houyi_studio.server.chat.types."""

from __future__ import annotations

import time

from houyi_studio.server.chat.types import (
    Attachment,
    Conversation,
    ConversationStatus,
    CreateConversationRequest,
    Message,
    MessageRole,
    SendMessageRequest,
    UpdateConversationRequest,
)

from houyi.adapters.llm.models import GPT_4O


class TestMessage:
    """Test Message model."""

    def test_defaults(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.reasoning_content is None
        assert msg.metadata == {}
        assert msg.message_id  # auto-generated
        assert msg.created_at > 0

    def test_to_llm_message(self):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there")
        llm = msg.to_llm_message()
        assert llm == {"role": "assistant", "content": "Hi there"}

    def test_to_llm_message_system(self):
        msg = Message(role=MessageRole.SYSTEM, content="Be helpful")
        llm = msg.to_llm_message()
        assert llm == {"role": "system", "content": "Be helpful"}

    def test_to_llm_message_tool(self):
        msg = Message(role=MessageRole.TOOL, content="result")
        llm = msg.to_llm_message()
        assert llm == {"role": "tool", "content": "result"}

    def test_assistant_tool_calls(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "houyi_find_files", "arguments": '{"query":"skill.md"}'},
                }
            ],
        )
        llm = msg.to_llm_message()
        assert llm["role"] == "assistant"
        assert llm["content"] == ""
        assert llm["tool_calls"] == msg.tool_calls

    def test_assistant_keeps_reasoning(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            reasoning_content="thinking...",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "houyi_find_files", "arguments": '{"query":"skill.md"}'},
                }
            ],
        )

        llm = msg.to_llm_message()

        assert llm["reasoning_content"] == "thinking..."
        assert llm["tool_calls"] == msg.tool_calls

    def test_tool_call_fields(self):
        msg = Message(
            role=MessageRole.TOOL,
            content='{"result":"ok"}',
            tool_call_id="call_1",
            name="houyi_find_files",
        )
        llm = msg.to_llm_message()
        assert llm == {
            "role": "tool",
            "content": '{"result":"ok"}',
            "tool_call_id": "call_1",
            "name": "houyi_find_files",
        }

    def test_serialization_roundtrip(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hello",
            reasoning_content="I think...",
            metadata={"tokens": 42},
        )
        data = msg.model_dump(mode="json")
        restored = Message(**data)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.reasoning_content == msg.reasoning_content
        assert restored.metadata == msg.metadata
        assert restored.message_id == msg.message_id

    def test_message_role_enum_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"
        assert len(MessageRole) == 4


class TestAttachment:
    """Test Attachment model."""

    def test_attachment_defaults(self):
        att = Attachment(
            filename="test.png", mime_type="image/png", data="data:image/png;base64,abc"
        )
        assert att.filename == "test.png"
        assert att.mime_type == "image/png"
        assert att.data == "data:image/png;base64,abc"
        assert att.size == 0  # default

    def test_attachment_with_size(self):
        att = Attachment(
            filename="photo.jpg",
            mime_type="image/jpeg",
            data="data:image/jpeg;base64,xyz",
            size=12345,
        )
        assert att.size == 12345

    def test_image_attachment(self):
        """Image attachments produce OpenAI multimodal content array."""
        att = Attachment(
            filename="img.png", mime_type="image/png", data="data:image/png;base64,AAAA"
        )
        msg = Message(role=MessageRole.USER, content="What is this?", attachments=[att])
        llm = msg.to_llm_message()
        assert llm["role"] == "user"
        assert isinstance(llm["content"], list)
        # Should have image_url part + text part
        types = [p["type"] for p in llm["content"]]
        assert "image_url" in types
        assert "text" in types
        # image_url part
        img_part = next(p for p in llm["content"] if p["type"] == "image_url")
        assert img_part["image_url"]["url"] == "data:image/png;base64,AAAA"
        # text part
        txt_part = next(p for p in llm["content"] if p["type"] == "text")
        assert txt_part["text"] == "What is this?"

    def test_multiple_images(self):
        """Multiple image attachments produce multiple image_url parts."""
        att1 = Attachment(filename="a.png", mime_type="image/png", data="data:image/png;base64,A")
        att2 = Attachment(filename="b.jpg", mime_type="image/jpeg", data="data:image/jpeg;base64,B")
        msg = Message(role=MessageRole.USER, content="Compare", attachments=[att1, att2])
        llm = msg.to_llm_message()
        img_parts = [p for p in llm["content"] if p["type"] == "image_url"]
        assert len(img_parts) == 2

    def test_image_only(self):
        """Image attachment with empty content — no text part."""
        att = Attachment(filename="img.png", mime_type="image/png", data="data:image/png;base64,X")
        msg = Message(role=MessageRole.USER, content="", attachments=[att])
        llm = msg.to_llm_message()
        types = [p["type"] for p in llm["content"]]
        assert "image_url" in types
        assert "text" not in types

    def test_plain_string(self):
        """No attachments → plain string content (not array)."""
        msg = Message(role=MessageRole.USER, content="Hello")
        llm = msg.to_llm_message()
        assert llm["content"] == "Hello"
        assert isinstance(llm["content"], str)

    def test_pdf_attachment_becomes_text(self):
        """PDF attachments produce a text description (no server-side parser)."""
        att = Attachment(
            filename="doc.pdf",
            mime_type="application/pdf",
            data="data:application/pdf;base64,X",
            size=1024,
        )
        msg = Message(role=MessageRole.USER, content="Read this", attachments=[att])
        llm = msg.to_llm_message()
        assert isinstance(llm["content"], str)
        assert "doc.pdf" in llm["content"]
        assert "content not extractable" in llm["content"]
        assert "Read this" in llm["content"]

    def test_message_text_file_extracted(self):
        """Text file attachments have their content extracted and included."""
        import base64

        text_content = "Hello from file"
        b64 = base64.b64encode(text_content.encode()).decode()
        att = Attachment(
            filename="readme.md",
            mime_type="text/markdown",
            data=f"data:text/markdown;base64,{b64}",
        )
        msg = Message(role=MessageRole.USER, content="Summarize", attachments=[att])
        llm = msg.to_llm_message()
        assert isinstance(llm["content"], str)
        assert "Hello from file" in llm["content"]
        assert "readme.md" in llm["content"]
        assert "Summarize" in llm["content"]

    def test_text_file_extension(self):
        """Text file detected by extension even with generic MIME."""
        import base64

        code = "print('hi')"
        b64 = base64.b64encode(code.encode()).decode()
        att = Attachment(
            filename="script.py",
            mime_type="application/octet-stream",
            data=f"data:application/octet-stream;base64,{b64}",
        )
        msg = Message(role=MessageRole.USER, content="Review", attachments=[att])
        llm = msg.to_llm_message()
        assert isinstance(llm["content"], str)
        assert "print('hi')" in llm["content"]

    def test_image_vision_false(self):
        """vision=False: image attachments become text placeholders."""
        att = Attachment(
            filename="img.png", mime_type="image/png", data="data:image/png;base64,AAAA"
        )
        msg = Message(role=MessageRole.USER, content="Describe", attachments=[att])
        llm = msg.to_llm_message(vision=False)
        assert isinstance(llm["content"], str)
        assert "[Image: img.png]" in llm["content"]
        assert "Describe" in llm["content"]

    def test_image_vision_true(self):
        """vision=True (default): image attachments produce image_url parts."""
        att = Attachment(
            filename="img.png", mime_type="image/png", data="data:image/png;base64,AAAA"
        )
        msg = Message(role=MessageRole.USER, content="Look", attachments=[att])
        llm = msg.to_llm_message(vision=True)
        assert isinstance(llm["content"], list)
        types = [p["type"] for p in llm["content"]]
        assert "image_url" in types

    def test_mixed_image_text(self):
        """Image + text file: image as image_url, text file extracted."""
        import base64

        img_att = Attachment(
            filename="photo.png", mime_type="image/png", data="data:image/png;base64,AAAA"
        )
        code = "x = 1"
        b64 = base64.b64encode(code.encode()).decode()
        txt_att = Attachment(
            filename="code.py", mime_type="text/x-python", data=f"data:text/x-python;base64,{b64}"
        )
        msg = Message(role=MessageRole.USER, content="Explain", attachments=[img_att, txt_att])
        llm = msg.to_llm_message(vision=True)
        assert isinstance(llm["content"], list)
        types = [p["type"] for p in llm["content"]]
        assert "image_url" in types
        assert "text" in types
        text_parts = [p["text"] for p in llm["content"] if p["type"] == "text"]
        combined = "\n".join(text_parts)
        assert "x = 1" in combined
        assert "Explain" in combined

    def test_attachment_serialization_roundtrip(self):
        """Attachment survives JSON serialization roundtrip."""
        att = Attachment(
            filename="test.png", mime_type="image/png", data="data:image/png;base64,abc", size=999
        )
        msg = Message(role=MessageRole.USER, content="Hi", attachments=[att])
        data = msg.model_dump(mode="json")
        restored = Message(**data)
        assert len(restored.attachments) == 1
        assert restored.attachments[0].filename == "test.png"
        assert restored.attachments[0].mime_type == "image/png"
        assert restored.attachments[0].data == "data:image/png;base64,abc"
        assert restored.attachments[0].size == 999

    def test_request_attachments(self):
        """SendMessageRequest accepts attachments."""
        att = Attachment(
            filename="img.png", mime_type="image/png", data="data:image/png;base64,X", size=100
        )
        req = SendMessageRequest(content="Look", attachments=[att])
        assert len(req.attachments) == 1
        assert req.attachments[0].filename == "img.png"

    def test_request_defaults(self):
        """SendMessageRequest defaults to empty attachments."""
        req = SendMessageRequest(content="Hello")
        assert req.attachments == []


class TestConversation:
    """Test Conversation model."""

    def test_defaults(self):
        conv = Conversation()
        assert conv.title == "New Chat"
        assert conv.status == ConversationStatus.ACTIVE
        assert conv.messages == []
        assert conv.model == ""
        assert conv.system_instructions == ""
        assert conv.stream is None
        assert conv.schema_version == 1
        assert conv.conversation_id  # auto-generated

    def test_message_count(self):
        conv = Conversation(
            messages=[
                Message(role=MessageRole.USER, content="Hi"),
                Message(role=MessageRole.ASSISTANT, content="Hello"),
            ]
        )
        assert conv.message_count == 2

    def test_message_count_empty(self):
        conv = Conversation()
        assert conv.message_count == 0

    def test_visible_count(self):
        conv = Conversation(
            messages=[
                Message(role=MessageRole.SYSTEM, content="sys"),
                Message(role=MessageRole.USER, content="Hi"),
                Message(role=MessageRole.TOOL, content="tool result"),
                Message(role=MessageRole.ASSISTANT, content="Hello"),
            ]
        )
        assert conv.visible_message_count == 2

    def test_last_message_at(self):
        t = time.time()
        conv = Conversation(
            messages=[
                Message(role=MessageRole.USER, content="Hi", created_at=t - 10),
                Message(role=MessageRole.ASSISTANT, content="Hello", created_at=t),
            ]
        )
        assert conv.last_message_at == t

    def test_last_message_at_empty(self):
        conv = Conversation()
        assert conv.last_message_at is None

    def test_stream_default_none(self):
        conv = Conversation()
        assert conv.stream is None

    def test_stream_explicit_true(self):
        conv = Conversation(stream=True)
        assert conv.stream is True

    def test_stream_explicit_false(self):
        conv = Conversation(stream=False)
        assert conv.stream is False

    def test_stream_in_summary(self):
        conv = Conversation(stream=False)
        summary = conv.to_summary()
        assert summary["stream"] is False

    def test_stream_none_in_summary(self):
        conv = Conversation()
        summary = conv.to_summary()
        assert summary["stream"] is None

    def test_stream_serialization_roundtrip(self):
        conv = Conversation(stream=False)
        data = conv.model_dump(mode="json")
        restored = Conversation(**data)
        assert restored.stream is False

    def test_to_summary(self):
        conv = Conversation(
            conversation_id="abc123",
            title="Test Chat",
            model=GPT_4O,
            messages=[Message(role=MessageRole.USER, content="Hi")],
        )
        summary = conv.to_summary()
        assert summary["conversation_id"] == "abc123"
        assert summary["title"] == "Test Chat"
        assert summary["status"] == "active"
        assert summary["message_count"] == 1
        assert summary["visible_message_count"] == 1
        assert summary["model"] == GPT_4O
        assert "created_at" in summary
        assert "updated_at" in summary
        assert "last_message_at" in summary
        assert summary["bookmarked"] is False
        assert "stream" in summary

    def test_to_summary_bookmarked(self):
        conv = Conversation(
            conversation_id="bm1",
            title="Bookmarked Chat",
            bookmarked=True,
        )
        summary = conv.to_summary()
        assert summary["bookmarked"] is True

    def test_bookmarked_default_false(self):
        conv = Conversation()
        assert conv.bookmarked is False

    def test_get_llm_messages(self):
        conv = Conversation(
            messages=[
                Message(role=MessageRole.SYSTEM, content="Be helpful"),
                Message(role=MessageRole.USER, content="Hi"),
                Message(role=MessageRole.ASSISTANT, content="Hello"),
            ]
        )
        llm_msgs = conv.get_llm_messages()
        assert len(llm_msgs) == 3
        assert llm_msgs[0] == {"role": "system", "content": "Be helpful"}
        assert llm_msgs[1] == {"role": "user", "content": "Hi"}
        assert llm_msgs[2] == {"role": "assistant", "content": "Hello"}

    def test_serialization_roundtrip(self):
        conv = Conversation(
            title="Round Trip",
            model="test-model",
            system_instructions="Be concise",
            messages=[
                Message(role=MessageRole.USER, content="Hi"),
            ],
            metadata={"key": "value"},
        )
        data = conv.model_dump(mode="json")
        restored = Conversation(**data)
        assert restored.title == conv.title
        assert restored.model == conv.model
        assert restored.system_instructions == conv.system_instructions
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Hi"
        assert restored.metadata == {"key": "value"}
        assert restored.schema_version == 1

    def test_roundtrip_compaction(self):
        conv = Conversation(
            metadata={
                "compaction_history": [
                    {
                        "compaction_id": "cmp_1",
                        "trigger": "manual",
                        "backup_id": "bck_1",
                        "pressure_level": "high",
                    }
                ]
            }
        )
        restored = Conversation(**conv.model_dump(mode="json"))
        history = restored.metadata["compaction_history"]
        assert history[0]["backup_id"] == "bck_1"
        assert history[0]["pressure_level"] == "high"

    def test_conversation_status_enum(self):
        assert ConversationStatus.ACTIVE.value == "active"
        assert ConversationStatus.ARCHIVED.value == "archived"


class TestRequestModels:
    """Test API request models."""

    def test_create_conversation_defaults(self):
        req = CreateConversationRequest()
        assert req.title == "New Chat"
        assert req.model == ""
        assert req.system_instructions == ""
        assert req.metadata == {}

    def test_send_message_required_content(self):
        req = SendMessageRequest(content="Hello")
        assert req.content == "Hello"
        assert req.model is None
        assert req.temperature is None
        assert req.max_tokens is None

    def test_send_message_accepts_deep_research_toggle(self):
        req = SendMessageRequest(content="Hello", enable_deep_research=True)
        assert req.enable_deep_research is True

    def test_update_conversation_all_none(self):
        req = UpdateConversationRequest()
        assert req.title is None
        assert req.status is None
        assert req.system_instructions is None
        assert req.model is None
        assert req.stream is None

    def test_update_conversation_stream_true(self):
        req = UpdateConversationRequest(stream=True)
        assert req.stream is True

    def test_update_conversation_stream_false(self):
        req = UpdateConversationRequest(stream=False)
        assert req.stream is False

    def test_update_stream_reset(self):
        req = UpdateConversationRequest.model_validate({"stream": None})
        assert req.stream is None
        raw = req.model_dump(exclude_unset=True)
        assert "stream" in raw


class TestMessageBookmark:
    """Test message-level bookmark functionality."""

    def test_default_false(self):
        """Message.bookmarked defaults to False."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.bookmarked is False

    def test_set_true(self):
        """Message.bookmarked can be set to True."""
        msg = Message(role=MessageRole.USER, content="Hello", bookmarked=True)
        assert msg.bookmarked is True

    def test_roundtrip_true(self):
        """bookmarked survives JSON serialization roundtrip."""
        msg = Message(role=MessageRole.USER, content="Hello", bookmarked=True)
        data = msg.model_dump(mode="json")
        assert data["bookmarked"] is True
        restored = Message(**data)
        assert restored.bookmarked is True

    def test_roundtrip_false(self):
        """bookmarked=False also survives roundtrip."""
        msg = Message(role=MessageRole.USER, content="Hello")
        data = msg.model_dump(mode="json")
        assert data["bookmarked"] is False
        restored = Message(**data)
        assert restored.bookmarked is False

    def test_conversation_messages(self):
        """Conversation can contain messages with mixed bookmark states."""
        msgs = [
            Message(role=MessageRole.USER, content="Q1", bookmarked=True),
            Message(role=MessageRole.ASSISTANT, content="A1"),
            Message(role=MessageRole.USER, content="Q2", bookmarked=True),
            Message(role=MessageRole.ASSISTANT, content="A2"),
        ]
        conv = Conversation(messages=msgs)
        bookmarked = [m for m in conv.messages if m.bookmarked]
        assert len(bookmarked) == 2
        assert bookmarked[0].content == "Q1"
        assert bookmarked[1].content == "Q2"


class TestTextExtraction:
    """Test text extraction from attachments in to_llm_message."""

    def test_text_file(self):
        """Text file content is decoded and included in LLM message."""
        import base64

        content = "def hello():\n    print('world')"
        b64 = base64.b64encode(content.encode()).decode()
        att = Attachment(
            filename="hello.py",
            mime_type="text/x-python",
            data=f"data:text/x-python;base64,{b64}",
        )
        msg = Message(role=MessageRole.USER, content="Review this", attachments=[att])
        llm = msg.to_llm_message()
        assert isinstance(llm["content"], str)
        assert "def hello():" in llm["content"]
        assert "print('world')" in llm["content"]
        assert "hello.py" in llm["content"]

    def test_json_mime(self):
        """JSON file detected by application/json MIME type."""
        import base64

        content = '{"key": "value"}'
        b64 = base64.b64encode(content.encode()).decode()
        att = Attachment(
            filename="data.json",
            mime_type="application/json",
            data=f"data:application/json;base64,{b64}",
        )
        msg = Message(role=MessageRole.USER, content="Parse", attachments=[att])
        llm = msg.to_llm_message()
        assert '"key": "value"' in llm["content"]

    def test_pdf_binary(self):
        """PDF (binary) falls back to filename description."""
        att = Attachment(
            filename="report.pdf",
            mime_type="application/pdf",
            data="data:application/pdf;base64,JVBERi0=",
            size=50000,
        )
        msg = Message(role=MessageRole.USER, content="Summarize", attachments=[att])
        llm = msg.to_llm_message()
        assert isinstance(llm["content"], str)
        assert "report.pdf" in llm["content"]
        assert "content not extractable" in llm["content"]

    def test_excel_fallback(self):
        """Excel file (binary) falls back to filename description."""
        att = Attachment(
            filename="data.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            data="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,UEsD",
            size=12000,
        )
        msg = Message(role=MessageRole.USER, content="Analyze", attachments=[att])
        llm = msg.to_llm_message()
        assert "data.xlsx" in llm["content"]
        assert "content not extractable" in llm["content"]

    def test_large_text_file_truncated(self):
        """Text files larger than 100K chars are truncated."""
        import base64

        content = "x" * 150_000
        b64 = base64.b64encode(content.encode()).decode()
        att = Attachment(
            filename="huge.txt",
            mime_type="text/plain",
            data=f"data:text/plain;base64,{b64}",
        )
        msg = Message(role=MessageRole.USER, content="Read", attachments=[att])
        llm = msg.to_llm_message()
        assert "truncated" in llm["content"]
        # Should not contain the full 150K chars
        assert len(llm["content"]) < 150_000

    def test_mixed_vision_true(self):
        """Image + text file with vision=True: image as image_url, text extracted."""
        import base64

        img = Attachment(
            filename="pic.png", mime_type="image/png", data="data:image/png;base64,AAAA"
        )
        code = "SELECT * FROM users"
        b64 = base64.b64encode(code.encode()).decode()
        sql = Attachment(
            filename="query.sql", mime_type="text/plain", data=f"data:text/plain;base64,{b64}"
        )
        msg = Message(role=MessageRole.USER, content="Explain", attachments=[img, sql])
        llm = msg.to_llm_message(vision=True)
        assert isinstance(llm["content"], list)
        types = [p["type"] for p in llm["content"]]
        assert "image_url" in types
        assert "text" in types
        text_content = " ".join(p["text"] for p in llm["content"] if p["type"] == "text")
        assert "SELECT * FROM users" in text_content

    def test_mixed_vision_false(self):
        """Image + text file with vision=False: image placeholder, text extracted."""
        import base64

        img = Attachment(
            filename="pic.png", mime_type="image/png", data="data:image/png;base64,AAAA"
        )
        code = "print('hi')"
        b64 = base64.b64encode(code.encode()).decode()
        py = Attachment(
            filename="script.py", mime_type="text/x-python", data=f"data:text/x-python;base64,{b64}"
        )
        msg = Message(role=MessageRole.USER, content="Review", attachments=[img, py])
        llm = msg.to_llm_message(vision=False)
        assert isinstance(llm["content"], str)
        assert "[Image: pic.png]" in llm["content"]
        assert "print('hi')" in llm["content"]
