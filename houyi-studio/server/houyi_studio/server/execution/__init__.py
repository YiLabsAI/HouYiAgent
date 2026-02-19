"""Execution engine subsystem.

Public API:
    - ExecutionEngine: Main execution orchestrator that manages plan execution lifecycle
    - ExecutionContext: Per-execution state container holding all runtime context
    - ExecutionCommandHandler: WebSocket command handler for execution lifecycle
      (start, pause, resume, abort, retry, restore)
    - LifecycleHook: Protocol for execution lifecycle callbacks
    - ObservationService: Event dispatch bridge (execution -> gateway)

Internal:
    - NodeExecutionFlow, LLMExecutionFlow: Node-level execution orchestration
    - NodeExecutors: Concrete node executor implementations (LLM, Tool, Verify, Route, Logic)
    - PlanExecutionLoop: Top-level plan execution loop
    - CheckpointService: Checkpoint creation and restoration
    - WorkflowService: Workflow persistence
    - Stores: ExecutionStore, CheckpointStore, PlanStore
"""
