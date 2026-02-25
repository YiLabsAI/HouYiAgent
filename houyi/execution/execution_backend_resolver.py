"""Resolver for selecting execution backends."""

from __future__ import annotations

import os

from houyi.config.env_config import ENV_EXECUTION_BACKEND
from houyi.execution.execution_backends import (
    DistributedBackend,
    ExecutionBackend,
    LocalBackend,
    SandboxBackend,
)


class ExecutionBackendResolver:
    """Resolve execution backend based on environment configuration."""

    def resolve(self) -> ExecutionBackend:
        backend_name = (os.getenv(ENV_EXECUTION_BACKEND) or "local").strip().lower()
        if backend_name == "distributed":
            return DistributedBackend()
        if backend_name == "sandbox":
            return SandboxBackend()
        return LocalBackend()
