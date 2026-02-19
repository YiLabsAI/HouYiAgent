"""Tool call subsystem.

Public API:
    - ToolCallService: Orchestrates tool call execution with SDK integration,
      consent management, and result formatting
    - ToolCallCoordinator: Manages tool call caches and service lifecycle

Internal:
    - ToolCallResponse: Tool call response assembly and streaming output
"""
