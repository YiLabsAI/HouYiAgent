"""Validate backend resolution from HOUYI_EXECUTION_BACKEND values."""

import os

from houyi.application.workflow.execution_backend_resolver import ExecutionBackendResolver
from houyi.application.workflow.execution_backends import (
    DistributedBackend,
    LocalBackend,
    SandboxBackend,
)


class TestExecutionBackendResolver:
    def test_defaults_to_local(self, monkeypatch):
        monkeypatch.delenv("HOUYI_EXECUTION_BACKEND", raising=False)
        backend = ExecutionBackendResolver().resolve()
        assert isinstance(backend, LocalBackend)

    def test_resolves_distributed(self, monkeypatch):
        monkeypatch.setenv("HOUYI_EXECUTION_BACKEND", "distributed")
        backend = ExecutionBackendResolver().resolve()
        assert isinstance(backend, DistributedBackend)

    def test_resolves_sandbox(self, monkeypatch):
        monkeypatch.setenv("HOUYI_EXECUTION_BACKEND", "sandbox")
        backend = ExecutionBackendResolver().resolve()
        assert isinstance(backend, SandboxBackend)

    def test_strips_and_lowercases(self, monkeypatch):
        monkeypatch.setenv("HOUYI_EXECUTION_BACKEND", "  LoCaL  ")
        backend = ExecutionBackendResolver().resolve()
        assert isinstance(backend, LocalBackend)
        assert os.getenv("HOUYI_EXECUTION_BACKEND") is not None
