"""Preprocessor pipeline for deterministic pre-LLM execution.

Preprocessors run *before* the LLM makes tool selection decisions.
They perform deterministic operations such as:
  - Script execution (shell commands)
  - Deterministic retrieval (file reads, pattern matches)
  - Context enrichment (inject information into messages)

Each preprocessor is defined in a skill's ``preprocessors[]`` field and
can produce ``PreprocessorResult`` objects that augment the tool-calling
context.

Design reference: §3.4, §4.4 of simpleskill-design.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PreprocessorType(str, Enum):
    """Supported preprocessor execution types."""

    COMMAND = "command"
    """Execute a shell command and capture stdout."""

    SCRIPT = "script"
    """Execute a Python script at a given path."""

    RETRIEVAL = "retrieval"
    """Run a deterministic retrieval query (e.g., grep, file read)."""


@dataclass
class PreprocessorSpec:
    """Definition of a single preprocessor step.

    Parsed from a skill's ``preprocessors[]`` manifest field.
    """

    type: PreprocessorType
    """Execution type."""

    command: str = ""
    """Shell command (type=command) or script path (type=script)."""

    query: str = ""
    """Retrieval query pattern (type=retrieval)."""

    target: str = ""
    """Target file/directory for retrieval operations."""

    timeout: float = 30.0
    """Maximum execution time in seconds."""

    inject_as: str = "system"
    """Where to inject results: 'system' (system message), 'context' (metadata)."""

    description: str = ""
    """Human-readable description for observability."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessorSpec:
        """Create a PreprocessorSpec from a dictionary (manifest parsing)."""
        ptype = data.get("type", "command")
        return cls(
            type=PreprocessorType(ptype)
            if ptype in PreprocessorType.__members__.values()
            else PreprocessorType.COMMAND,
            command=data.get("command", ""),
            query=data.get("query", ""),
            target=data.get("target", ""),
            timeout=float(data.get("timeout", 30.0)),
            inject_as=data.get("inject_as", "system"),
            description=data.get("description", ""),
        )


@dataclass
class PreprocessorResult:
    """Output from a preprocessor execution."""

    preprocessor: PreprocessorSpec
    """The spec that produced this result."""

    success: bool = True
    """Whether the preprocessor executed successfully."""

    output: str = ""
    """Captured output (stdout for commands, content for retrieval)."""

    error: str = ""
    """Error message if execution failed."""

    elapsed_ms: float = 0.0
    """Wall-clock execution time in milliseconds."""


class PreprocessorPipeline:
    """Execute an ordered list of preprocessors before LLM invocation.

    Usage::

        pipeline = PreprocessorPipeline(specs)
        results = await pipeline.run(working_dir="/path/to/skill")
        enriched_messages = pipeline.inject(messages, results)
    """

    def __init__(self, specs: list[PreprocessorSpec]) -> None:
        self._specs = list(specs)

    @property
    def specs(self) -> list[PreprocessorSpec]:
        """Read-only access to the preprocessor specifications."""
        return list(self._specs)

    async def run(
        self,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[PreprocessorResult]:
        """Execute all preprocessors sequentially and return results.

        Args:
            working_dir: Working directory for command/script execution.
            env: Additional environment variables for subprocess execution.

        Returns:
            Ordered list of results, one per preprocessor spec.
        """
        results: list[PreprocessorResult] = []
        for spec in self._specs:
            result = await self._execute_one(spec, working_dir, env)
            results.append(result)
            if not result.success:
                logger.warning(
                    "Preprocessor failed (type=%s, desc=%s): %s",
                    spec.type,
                    spec.description,
                    result.error,
                )
        return results

    async def _execute_one(
        self,
        spec: PreprocessorSpec,
        working_dir: str | None,
        env: dict[str, str] | None,
    ) -> PreprocessorResult:
        """Execute a single preprocessor step."""
        import time

        start = time.perf_counter()
        try:
            if spec.type == PreprocessorType.COMMAND:
                output = await self._run_command(spec, working_dir, env)
            elif spec.type == PreprocessorType.SCRIPT:
                output = await self._run_script(spec, working_dir, env)
            elif spec.type == PreprocessorType.RETRIEVAL:
                output = await self._run_retrieval(spec, working_dir)
            else:
                return PreprocessorResult(
                    preprocessor=spec,
                    success=False,
                    error=f"Unknown preprocessor type: {spec.type}",
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )
            elapsed = (time.perf_counter() - start) * 1000
            return PreprocessorResult(
                preprocessor=spec,
                success=True,
                output=output,
                elapsed_ms=elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            return PreprocessorResult(
                preprocessor=spec,
                success=False,
                error=f"Timeout after {spec.timeout}s",
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return PreprocessorResult(
                preprocessor=spec,
                success=False,
                error=str(exc),
                elapsed_ms=elapsed,
            )

    async def _run_command(
        self,
        spec: PreprocessorSpec,
        working_dir: str | None,
        env: dict[str, str] | None,
    ) -> str:
        """Execute a shell command and return stdout."""
        full_env = {**os.environ, **(env or {})}
        process = await asyncio.create_subprocess_shell(
            spec.command,
            cwd=working_dir,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=spec.timeout)
        if process.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Command exited with code {process.returncode}: {err_text}")
        return stdout.decode("utf-8", errors="replace").strip()

    async def _run_script(
        self,
        spec: PreprocessorSpec,
        working_dir: str | None,
        env: dict[str, str] | None,
    ) -> str:
        """Execute a Python script and return its stdout."""
        import sys

        script_path = spec.command
        if working_dir and not os.path.isabs(script_path):
            script_path = os.path.join(working_dir, script_path)
        full_env = {**os.environ, **(env or {})}
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            cwd=working_dir,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=spec.timeout)
        if process.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Script exited with code {process.returncode}: {err_text}")
        return stdout.decode("utf-8", errors="replace").strip()

    async def _run_retrieval(
        self,
        spec: PreprocessorSpec,
        working_dir: str | None,
    ) -> str:
        """Execute a deterministic retrieval operation (file read or grep)."""
        target = spec.target
        if working_dir and not os.path.isabs(target):
            target = os.path.join(working_dir, target)

        if spec.query:
            # Grep-style retrieval: search for pattern in target
            return await self._grep_retrieval(spec.query, target, spec.timeout)
        else:
            # File read retrieval
            return await self._file_retrieval(target)

    async def _file_retrieval(self, path: str) -> str:
        """Read a file's contents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_file_sync, path)

    @staticmethod
    def _read_file_sync(path: str) -> str:
        """Synchronous file read helper."""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    async def _grep_retrieval(self, pattern: str, target: str, timeout: float) -> str:
        """Search for a pattern in a file/directory using grep."""
        cmd = f"grep -rn {_shell_quote(pattern)} {_shell_quote(target)}"
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return stdout.decode("utf-8", errors="replace").strip()

    @staticmethod
    def inject(
        messages: list[dict[str, Any]],
        results: list[PreprocessorResult],
    ) -> list[dict[str, Any]]:
        """Inject successful preprocessor outputs into the message list.

        For ``inject_as='system'``: Appends a system message with the output.
        For ``inject_as='context'``: Appends as a structured context message.

        Args:
            messages: The current message list (will not be mutated).
            results: Preprocessor results to inject.

        Returns:
            A new list of messages with preprocessor outputs injected.
        """
        injections: list[dict[str, Any]] = []
        for result in results:
            if not result.success or not result.output:
                continue
            label = (
                result.preprocessor.description or f"preprocessor:{result.preprocessor.type.value}"
            )
            if result.preprocessor.inject_as == "system":
                injections.append(
                    {
                        "role": "system",
                        "content": f"[{label}]\n{result.output}",
                    }
                )
            else:
                injections.append(
                    {
                        "role": "system",
                        "content": f"[context:{label}]\n{result.output}",
                    }
                )

        if not injections:
            return messages

        # Insert preprocessor results after any existing system messages
        # but before the first non-system message.
        new_messages = list(messages)
        insert_idx = 0
        for i, msg in enumerate(new_messages):
            if msg.get("role") != "system":
                insert_idx = i
                break
        else:
            insert_idx = len(new_messages)

        for j, injection in enumerate(injections):
            new_messages.insert(insert_idx + j, injection)

        return new_messages


def _shell_quote(s: str) -> str:
    """Simple shell quoting for safe command construction."""
    return "'" + s.replace("'", "'\\''") + "'"
