"""Unit tests for execution/execution_backends.py."""

import pytest

from houyi.execution.execution_backends import DistributedBackend, LocalBackend, SandboxBackend


@pytest.mark.asyncio
async def test_local_backend_creates_task():
    backend = LocalBackend()

    async def runner(_session_id, _execution, _plan):
        return "ok"

    task = await backend.start("s1", execution=None, plan=None, runner=runner)
    assert task is not None
    assert await task == "ok"


@pytest.mark.asyncio
async def test_distributed_backend_not_implemented():
    backend = DistributedBackend()
    with pytest.raises(NotImplementedError):
        await backend.start("s1", execution=None, plan=None, runner=lambda *_args: None)


@pytest.mark.asyncio
async def test_sandbox_backend_not_implemented():
    backend = SandboxBackend()
    with pytest.raises(NotImplementedError):
        await backend.start("s1", execution=None, plan=None, runner=lambda *_args: None)
