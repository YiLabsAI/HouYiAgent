"""Resource-management command handler (Workflow, Knowledge, Document).

Part of the command-handler hierarchy extracted from app.py following the
**Single Responsibility Principle (SRP)** and **Open/Closed Principle (OCP)**:

Architecture
~~~~~~~~~~~~
::

    WebSocket message
         │
         ▼
    CommandParser          → parse raw JSON into typed/dict command
         │
         ▼
    CommandDispatcher      → route command_type to the correct handler  (OCP)
         │
         ├─► SkillCommandHandler          – skill lifecycle (load/unload/configure/dry-run)
         ├─► CommandHandler    ◄── this module  – resource CRUD (workflow/knowledge/document)
         └─► ExecutionCommandHandler      – execution lifecycle (start/pause/abort/patch/restore)

Design rationale
~~~~~~~~~~~~~~~~
*   **SRP**: This class owns *resource-management* commands — operations that
    create, read, update, or delete persistent artefacts (workflows, knowledge
    libraries, documents, chunks) and utility commands (frontend log bridging,
    log-level adjustment).  It does *not* touch execution state or skill
    lifecycle, which belong to their respective handlers.

*   **OCP**: New resource domains (e.g., a future "prompt-template" CRUD) can be
    added here — or in a new handler — without modifying the dispatcher or
    other handlers.

*   **Dependency Inversion**: The class receives collaborators (event sender,
    execution engine, knowledge service) as constructor-injected callables,
    making it fully testable without spinning up a real server.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from houyi.interface.protocol.ir import PlanIR

from ..logging_config import get_log_level, set_log_level
from .events import LogLevelEvent


class CommandHandler:
    """Resource-management command handler for workflow, knowledge-base, and document operations.

    Responsibilities
    ----------------
    - Workflow persistence: save / load / list workflows via ExecutionEngine.workflow_service.
    - Knowledge-base lifecycle: CRUD on knowledge libraries, file ingestion (with progress
      streaming), index rebuilds, and cancellation.
    - Document management: list / get / delete / enable / disable documents and chunk
      preview within a knowledge library.
    - Utility: bridge frontend log messages to the backend logger; synchronise the
      server-wide log level on client request.

    Integration
    -----------
    Registered with CommandDispatcher for all SUPPORTED_COMMAND_TYPES.
    The dispatcher calls can_handle → handle on each incoming dict-based
    command whose command_type matches.
    """

    SUPPORTED_COMMAND_TYPES = {
        "save_workflow",
        "load_workflow",
        "list_workflows",
        "list_knowledge_libraries",
        "create_knowledge_library",
        "delete_knowledge_library",
        "update_knowledge_library",
        "search_knowledge",
        "ingest_knowledge_files",
        "rebuild_knowledge_index",
        "cancel_ingest",
        "list_documents",
        "get_document",
        "delete_document",
        "disable_document",
        "enable_document",
        "list_chunks",
        "preview_chunks",
        "frontend_log",
        "set_log_level",
    }

    def __init__(
        self,
        *,
        send_event: Callable[[str, object], Awaitable[None]],
        get_execution_engine: Callable[[], object],
        sanitize_plan_payload: Callable[[dict[str, Any]], dict[str, Any]],
        knowledge_service_getter: Callable[[], object],
        logger: logging.Logger | None = None,
    ) -> None:
        self._send_event = send_event
        self._get_execution_engine = get_execution_engine
        self._sanitize_plan_payload = sanitize_plan_payload
        self._knowledge_service_getter = knowledge_service_getter
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def can_handle(cls, command: object) -> bool:
        return (
            isinstance(command, dict)
            and isinstance(command.get("command_type"), str)
            and command["command_type"] in cls.SUPPORTED_COMMAND_TYPES
        )

    async def handle(self, command: dict[str, Any], session_id: str) -> None:
        command_type = command.get("command_type")
        if command_type == "save_workflow":
            await self._handle_save_workflow(command, session_id)
        elif command_type == "load_workflow":
            await self._handle_load_workflow(command, session_id)
        elif command_type == "list_workflows":
            await self._handle_list_workflows(session_id)
        elif command_type == "list_knowledge_libraries":
            await self._handle_list_knowledge_libraries(session_id)
        elif command_type == "create_knowledge_library":
            await self._handle_create_knowledge_library(command, session_id)
        elif command_type == "delete_knowledge_library":
            await self._handle_delete_knowledge_library(command, session_id)
        elif command_type == "update_knowledge_library":
            await self._handle_update_knowledge_library(command, session_id)
        elif command_type == "search_knowledge":
            await self._handle_search_knowledge(command, session_id)
        elif command_type == "ingest_knowledge_files":
            await self._handle_ingest_knowledge_files(command, session_id)
        elif command_type == "rebuild_knowledge_index":
            await self._handle_rebuild_knowledge_index(command, session_id)
        elif command_type == "cancel_ingest":
            await self._handle_cancel_ingest(command)
        elif command_type == "list_documents":
            await self._handle_list_documents(command, session_id)
        elif command_type == "get_document":
            await self._handle_get_document(command, session_id)
        elif command_type == "delete_document":
            await self._handle_delete_document(command, session_id)
        elif command_type == "disable_document":
            await self._handle_disable_document(command, session_id)
        elif command_type == "enable_document":
            await self._handle_enable_document(command, session_id)
        elif command_type == "list_chunks":
            await self._handle_list_chunks(command, session_id)
        elif command_type == "preview_chunks":
            await self._handle_preview_chunks(command, session_id)
        elif command_type == "frontend_log":
            self._handle_frontend_log(command)
        elif command_type == "set_log_level":
            await self._handle_set_log_level(command, session_id)

    @staticmethod
    def _event_id() -> str:
        return f"evt_{uuid4().hex[:8]}"

    async def _handle_save_workflow(self, command: dict[str, Any], session_id: str) -> None:
        engine = self._get_execution_engine()
        workflow_name = command.get("workflow_name")
        self._logger.debug("=== Save workflow command received ===")
        self._logger.debug("Workflow name: %s", workflow_name)
        self._logger.debug("Session ID: %s", session_id)

        plan_payload = command.get("plan")
        current_plan = None
        if plan_payload:
            try:
                sanitized_payload = self._sanitize_plan_payload(plan_payload)
                current_plan = PlanIR.model_validate(sanitized_payload)
                self._logger.debug("Using plan payload from client for workflow save")
            except Exception:
                self._logger.warning(
                    "Failed to parse plan payload for workflow save; falling back to session plan",
                    exc_info=True,
                )
        if current_plan is None:
            current_plan = engine.plan_service.get_current_plan(session_id)
        self._logger.debug("Current plan exists: %s", current_plan is not None)

        if current_plan:
            self._logger.info(
                "Plan has %d nodes, %d edges", len(current_plan.nodes), len(current_plan.edges)
            )
            success = engine.workflow_service.save_workflow(workflow_name, current_plan)
            if success:
                self._logger.info("✓ Workflow '%s' saved successfully", workflow_name)
            else:
                self._logger.error("✗ Failed to save workflow '%s'", workflow_name)
        else:
            self._logger.warning("✗ No plan to save for session: %s", session_id)
            self._logger.warning("Available sessions: %s", list(engine.plans.keys()))

    async def _handle_load_workflow(self, command: dict[str, Any], session_id: str) -> None:
        from .events import PlanUpdatedEvent

        engine = self._get_execution_engine()
        workflow_name = command.get("workflow_name")
        self._logger.debug("=== Load workflow command received ===")
        self._logger.debug("Workflow name: %s", workflow_name)
        self._logger.debug("Session ID: %s", session_id)

        plan = engine.workflow_service.load_workflow(workflow_name)
        if not plan:
            self._logger.error("✗ Failed to load workflow '%s'", workflow_name)
            return

        engine.plan_service.set_current_plan(session_id, plan, persist=True)
        plan_event = PlanUpdatedEvent(
            event_id=self._event_id(),
            session_id=session_id,
            plan=plan,
        )
        await self._send_event(session_id, plan_event)
        self._logger.info("✓ Workflow '%s' loaded and sent to frontend", workflow_name)

    async def _handle_list_workflows(self, session_id: str) -> None:
        from .events import WorkflowListEvent

        engine = self._get_execution_engine()
        self._logger.debug("=== List workflows command received ===")
        workflows = engine.workflow_service.list_workflows()
        self._logger.debug("Found %d workflows", len(workflows))
        workflow_event = WorkflowListEvent(
            event_id=self._event_id(),
            session_id=session_id,
            workflows=workflows,
        )
        await self._send_event(session_id, workflow_event)

    async def _handle_list_knowledge_libraries(self, session_id: str) -> None:
        from .events import KnowledgeLibraryListEvent

        knowledge_service = self._knowledge_service_getter()
        libraries = knowledge_service.list_libraries()
        event = KnowledgeLibraryListEvent(
            event_id=self._event_id(),
            session_id=session_id,
            libraries=libraries,
        )
        await self._send_event(session_id, event)

    async def _handle_create_knowledge_library(
        self, command: dict[str, Any], session_id: str
    ) -> None:
        from .events import KnowledgeLibraryCreatedEvent

        knowledge_service = self._knowledge_service_getter()
        metadata = command.get("metadata") or {}
        if command.get("strategies"):
            metadata["strategies"] = command.get("strategies")
        if command.get("embedding_provider"):
            metadata["embedding_provider"] = command.get("embedding_provider")
        if command.get("contextual_retrieval") is not None:
            metadata["contextual_retrieval"] = command.get("contextual_retrieval")

        library = knowledge_service.create_library(
            name=command.get("name", "Untitled"),
            description=command.get("description", ""),
            mode=command.get("mode", "auto"),
            knowledge_dir=command.get("knowledge_dir", "./knowledge"),
            metadata=metadata,
        )
        event = KnowledgeLibraryCreatedEvent(
            event_id=self._event_id(),
            session_id=session_id,
            library=library,
        )
        await self._send_event(session_id, event)

    async def _handle_delete_knowledge_library(
        self, command: dict[str, Any], session_id: str
    ) -> None:
        from .events import KnowledgeErrorEvent, KnowledgeLibraryDeletedEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        if not library_id:
            return
        success = knowledge_service.delete_library(library_id)
        if success:
            event = KnowledgeLibraryDeletedEvent(
                event_id=self._event_id(),
                session_id=session_id,
                library_id=library_id,
            )
            await self._send_event(session_id, event)
            return
        event = KnowledgeErrorEvent(
            event_id=self._event_id(),
            session_id=session_id,
            error=f"Library not found: {library_id}",
            operation="delete",
        )
        await self._send_event(session_id, event)

    async def _handle_update_knowledge_library(
        self, command: dict[str, Any], session_id: str
    ) -> None:
        from .events import KnowledgeErrorEvent, KnowledgeLibraryUpdatedEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        updates = command.get("updates", {})
        if not library_id:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="library_id is required",
                    operation="update",
                ),
            )
            return

        library = knowledge_service.update_library(library_id, updates)
        if library:
            await self._send_event(
                session_id,
                KnowledgeLibraryUpdatedEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library=library,
                ),
            )
            return

        await self._send_event(
            session_id,
            KnowledgeErrorEvent(
                event_id=self._event_id(),
                session_id=session_id,
                error=f"Library not found: {library_id}",
                operation="update",
            ),
        )

    async def _handle_search_knowledge(self, command: dict[str, Any], session_id: str) -> None:
        from .events import KnowledgeErrorEvent, KnowledgeSearchResultsEvent

        knowledge_service = self._knowledge_service_getter()
        query = command.get("query", "")
        if not query:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="Query is required",
                    operation="search",
                ),
            )
            return

        results = await knowledge_service.search_knowledge(
            query=query,
            library_id=command.get("library_id"),
            mode=command.get("mode"),
            top_k=command.get("top_k", 10),
        )
        await self._send_event(
            session_id,
            KnowledgeSearchResultsEvent(
                event_id=self._event_id(),
                session_id=session_id,
                query=results.get("query", query),
                library_id=results.get("library_id", ""),
                results=results.get("results", []),
                mode_used=results.get("mode_used", ""),
                total_results=results.get("total_results", 0),
                quality=results.get("quality"),
            ),
        )

    async def _handle_ingest_knowledge_files(
        self, command: dict[str, Any], session_id: str
    ) -> None:
        from .events import (
            KnowledgeErrorEvent,
            KnowledgeIngestCompleteEvent,
            KnowledgeIngestProgressEvent,
            KnowledgeLibraryUpdatedEvent,
        )

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        paths = command.get("paths", [])
        if not library_id:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="library_id is required",
                    operation="ingest",
                ),
            )
            return
        if not paths:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="paths is required",
                    operation="ingest",
                ),
            )
            return

        async def progress_callback(
            progress: float, current_file: str, files_processed: int, total_files: int
        ) -> None:
            await self._send_event(
                session_id,
                KnowledgeIngestProgressEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library_id=library_id,
                    progress=progress,
                    current_file=current_file,
                    files_processed=files_processed,
                    total_files=total_files,
                ),
            )

        result = await knowledge_service.ingest_files(
            library_id=library_id,
            paths=paths,
            progress_callback=progress_callback,
        )
        # Build message: prioritise warning (degraded) over generic "success"
        if not result.get("success"):
            message = result.get("error", "Unknown error")
        elif result.get("warning"):
            message = result["warning"]
        else:
            message = "Ingest complete"

        await self._send_event(
            session_id,
            KnowledgeIngestCompleteEvent(
                event_id=self._event_id(),
                session_id=session_id,
                library_id=library_id,
                success=result.get("success", False),
                stats=result.get("stats", {}),
                message=message,
                warning=result.get("warning"),
            ),
        )
        library = knowledge_service.get_library(library_id)
        if library:
            await self._send_event(
                session_id,
                KnowledgeLibraryUpdatedEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library=library,
                ),
            )

    async def _handle_rebuild_knowledge_index(
        self, command: dict[str, Any], session_id: str
    ) -> None:
        from .events import (
            KnowledgeErrorEvent,
            KnowledgeIngestCompleteEvent,
            KnowledgeIngestProgressEvent,
            KnowledgeLibraryUpdatedEvent,
        )

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        if not library_id:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="library_id is required",
                    operation="rebuild",
                ),
            )
            return
        library = knowledge_service.get_library(library_id)
        if not library:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error=f"Library {library_id} not found",
                    operation="rebuild",
                ),
            )
            return

        async def progress_callback(
            progress: float, current_file: str, files_processed: int, total_files: int
        ) -> None:
            await self._send_event(
                session_id,
                KnowledgeIngestProgressEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library_id=library_id,
                    progress=progress,
                    current_file=current_file,
                    files_processed=files_processed,
                    total_files=total_files,
                ),
            )

        knowledge_dir = library.get("knowledge_dir", "./knowledge")
        upload_dir = knowledge_service.library_upload_dir(library_id)
        incremental = command.get("incremental", False)
        paths_to_index = [knowledge_dir]
        if upload_dir.exists():
            paths_to_index.append(str(upload_dir))

        result = await knowledge_service.ingest_files(
            library_id=library_id,
            paths=paths_to_index,
            progress_callback=progress_callback,
            incremental=incremental,
        )
        stats = result.get("stats", {})
        if not result.get("success"):
            message = result.get("error", "Unknown error")
        elif result.get("warning"):
            # Degraded: files imported but no embedding
            message = result["warning"]
        else:
            files_processed = stats.get("files_processed", 0)
            chunks_created = stats.get("chunks_created", 0)
            if files_processed > 0:
                message = f"Indexed {files_processed} files, {chunks_created} chunks"
            else:
                message = "No changes detected"

        await self._send_event(
            session_id,
            KnowledgeIngestCompleteEvent(
                event_id=self._event_id(),
                session_id=session_id,
                library_id=library_id,
                success=result.get("success", False),
                stats=stats,
                message=message,
                warning=result.get("warning"),
            ),
        )
        updated_library = knowledge_service.get_library(library_id)
        if updated_library:
            await self._send_event(
                session_id,
                KnowledgeLibraryUpdatedEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library=updated_library,
                ),
            )

    async def _handle_cancel_ingest(self, command: dict[str, Any]) -> None:
        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        if library_id:
            knowledge_service.cancel_ingest(library_id)

    async def _handle_list_documents(self, command: dict[str, Any], session_id: str) -> None:
        from .events import DocumentListEvent, KnowledgeErrorEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        if not library_id:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="library_id is required",
                    operation="list_documents",
                ),
            )
            return
        documents = knowledge_service.list_documents(library_id)
        await self._send_event(
            session_id,
            DocumentListEvent(
                event_id=self._event_id(),
                session_id=session_id,
                library_id=library_id,
                documents=documents,
            ),
        )

    async def _handle_get_document(self, command: dict[str, Any], session_id: str) -> None:
        from .events import DocumentDetailEvent, KnowledgeErrorEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        doc_id = command.get("doc_id")
        if not library_id or not doc_id:
            await self._send_event(
                session_id,
                KnowledgeErrorEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    error="library_id and doc_id are required",
                    operation="get_document",
                ),
            )
            return
        document = knowledge_service.get_document(library_id, doc_id)
        if document:
            await self._send_event(
                session_id,
                DocumentDetailEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library_id=library_id,
                    document=document,
                ),
            )

    async def _handle_delete_document(self, command: dict[str, Any], session_id: str) -> None:
        from .events import DocumentDeletedEvent, KnowledgeLibraryUpdatedEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        doc_id = command.get("doc_id")
        if not library_id or not doc_id:
            return
        success = knowledge_service.delete_document(library_id, doc_id)
        if not success:
            return
        await self._send_event(
            session_id,
            DocumentDeletedEvent(
                event_id=self._event_id(),
                session_id=session_id,
                library_id=library_id,
                doc_id=doc_id,
            ),
        )
        updated_library = knowledge_service.get_library(library_id)
        if updated_library:
            await self._send_event(
                session_id,
                KnowledgeLibraryUpdatedEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library=updated_library,
                ),
            )

    async def _handle_disable_document(self, command: dict[str, Any], session_id: str) -> None:
        from .events import DocumentStatusChangedEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        doc_id = command.get("doc_id")
        if not library_id or not doc_id:
            return
        document = knowledge_service.disable_document(library_id, doc_id)
        if document:
            await self._send_event(
                session_id,
                DocumentStatusChangedEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library_id=library_id,
                    doc_id=doc_id,
                    status="disabled",
                ),
            )

    async def _handle_enable_document(self, command: dict[str, Any], session_id: str) -> None:
        from .events import DocumentStatusChangedEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        doc_id = command.get("doc_id")
        if not library_id or not doc_id:
            return
        document = knowledge_service.enable_document(library_id, doc_id)
        if document:
            await self._send_event(
                session_id,
                DocumentStatusChangedEvent(
                    event_id=self._event_id(),
                    session_id=session_id,
                    library_id=library_id,
                    doc_id=doc_id,
                    status="indexed",
                ),
            )

    async def _handle_list_chunks(self, command: dict[str, Any], session_id: str) -> None:
        from .events import ChunkListEvent

        knowledge_service = self._knowledge_service_getter()
        library_id = command.get("library_id")
        doc_id = command.get("doc_id")
        if not library_id or not doc_id:
            return
        chunks = knowledge_service.list_chunks(library_id, doc_id)
        await self._send_event(
            session_id,
            ChunkListEvent(
                event_id=self._event_id(),
                session_id=session_id,
                library_id=library_id,
                doc_id=doc_id,
                chunks=chunks,
            ),
        )

    async def _handle_preview_chunks(self, command: dict[str, Any], session_id: str) -> None:
        from .events import ChunkPreviewEvent

        knowledge_service = self._knowledge_service_getter()
        content = command.get("content", "")
        chunk_size = command.get("chunk_size", 512)
        chunk_overlap = command.get("chunk_overlap", 50)
        strategy = command.get("strategy", "recursive")
        chunks = knowledge_service.preview_chunks(
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
        await self._send_event(
            session_id,
            ChunkPreviewEvent(
                event_id=self._event_id(),
                session_id=session_id,
                chunks=chunks,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy,
            ),
        )

    def _handle_frontend_log(self, command: dict[str, Any]) -> None:
        level = command.get("level", "info")
        category = command.get("category", "Frontend")
        message = command.get("message", "")
        log_prefix = f"[Frontend/{category}] {message}"
        if level == "debug":
            self._logger.debug(log_prefix)
        elif level == "warn":
            self._logger.warning(log_prefix)
        elif level == "error":
            self._logger.error(log_prefix)
        else:
            self._logger.debug(log_prefix)

    async def _handle_set_log_level(self, command: dict[str, Any], session_id: str) -> None:
        requested = command.get("level")
        if not isinstance(requested, str):
            return
        resolved_level = set_log_level(requested)
        self._logger.info(
            "Log level updated: requested=%s resolved=%s effective=%s",
            requested,
            resolved_level,
            get_log_level(),
        )
        await self._send_event(
            session_id,
            LogLevelEvent(
                event_id=self._event_id(),
                session_id=session_id,
                level=resolved_level.lower(),
                requested_level=requested,
            ),
        )
