"""Tests for the Preprocessor pipeline (M8).

Covers:
  - PreprocessorSpec.from_dict parsing
  - PreprocessorPipeline execution (command, retrieval, script)
  - PreprocessorPipeline.inject message augmentation
  - Error handling: timeout, invalid command, missing files
  - Edge cases: empty pipeline, all-failing preprocessors
"""

import os
import tempfile

import pytest

from houyi.domain.skill.preprocessor import (
    PreprocessorPipeline,
    PreprocessorResult,
    PreprocessorSpec,
    PreprocessorType,
)


class TestPreprocessorSpec:
    def test_from_dict_command(self):
        spec = PreprocessorSpec.from_dict(
            {
                "type": "command",
                "command": "echo hello",
                "timeout": 10,
                "inject_as": "system",
                "description": "greeting",
            }
        )
        assert spec.type == PreprocessorType.COMMAND
        assert spec.command == "echo hello"
        assert spec.timeout == 10.0
        assert spec.inject_as == "system"
        assert spec.description == "greeting"

    def test_from_dict_retrieval(self):
        spec = PreprocessorSpec.from_dict(
            {
                "type": "retrieval",
                "query": "TODO",
                "target": "/tmp/code",
            }
        )
        assert spec.type == PreprocessorType.RETRIEVAL
        assert spec.query == "TODO"
        assert spec.target == "/tmp/code"

    def test_from_dict_defaults(self):
        spec = PreprocessorSpec.from_dict({})
        assert spec.type == PreprocessorType.COMMAND
        assert spec.command == ""
        assert spec.timeout == 30.0
        assert spec.inject_as == "system"

    def test_unknown_type_fallback(self):
        spec = PreprocessorSpec.from_dict({"type": "unknown_type"})
        assert spec.type == PreprocessorType.COMMAND


class TestPreprocessorPipeline:
    @pytest.mark.asyncio
    async def test_command_success(self):
        spec = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="echo hello world",
        )
        pipeline = PreprocessorPipeline([spec])
        results = await pipeline.run()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "hello world"
        assert results[0].elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_command_failure(self):
        spec = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="exit 42",
        )
        pipeline = PreprocessorPipeline([spec])
        results = await pipeline.run()
        assert len(results) == 1
        assert results[0].success is False
        assert "42" in results[0].error

    @pytest.mark.asyncio
    async def test_command_timeout(self):
        spec = PreprocessorSpec(
            type=PreprocessorType.COMMAND,
            command="sleep 60",
            timeout=0.1,
        )
        pipeline = PreprocessorPipeline([spec])
        results = await pipeline.run()
        assert len(results) == 1
        assert results[0].success is False
        assert "Timeout" in results[0].error

    @pytest.mark.asyncio
    async def test_retrieval_file_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content 123")
            f.flush()
            path = f.name

        try:
            spec = PreprocessorSpec(
                type=PreprocessorType.RETRIEVAL,
                target=path,
            )
            pipeline = PreprocessorPipeline([spec])
            results = await pipeline.run()
            assert len(results) == 1
            assert results[0].success is True
            assert "test content 123" in results[0].output
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_retrieval_missing_file(self):
        spec = PreprocessorSpec(
            type=PreprocessorType.RETRIEVAL,
            target="/nonexistent/path/file.txt",
        )
        pipeline = PreprocessorPipeline([spec])
        results = await pipeline.run()
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error  # Should have an error message

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        pipeline = PreprocessorPipeline([])
        results = await pipeline.run()
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_preprocessors(self):
        specs = [
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="echo step1"),
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="echo step2"),
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="echo step3"),
        ]
        pipeline = PreprocessorPipeline(specs)
        results = await pipeline.run()
        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].output == "step1"
        assert results[1].output == "step2"
        assert results[2].output == "step3"

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        specs = [
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="echo ok"),
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="exit 1"),
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="echo after_fail"),
        ]
        pipeline = PreprocessorPipeline(specs)
        results = await pipeline.run()
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True  # Continues after failure

    def test_specs_property(self):
        specs = [
            PreprocessorSpec(type=PreprocessorType.COMMAND, command="echo a"),
        ]
        pipeline = PreprocessorPipeline(specs)
        assert len(pipeline.specs) == 1
        assert pipeline.specs[0].command == "echo a"


class TestPreprocessorInject:
    def test_inject_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        results = [
            PreprocessorResult(
                preprocessor=PreprocessorSpec(
                    type=PreprocessorType.COMMAND,
                    description="context loader",
                ),
                success=True,
                output="loaded context data",
            ),
        ]
        new_messages = PreprocessorPipeline.inject(messages, results)
        assert len(new_messages) == 3
        assert new_messages[1]["role"] == "system"
        assert "[context loader]" in new_messages[1]["content"]
        assert "loaded context data" in new_messages[1]["content"]

    def test_inject_context_mode(self):
        messages = [{"role": "user", "content": "Hello"}]
        results = [
            PreprocessorResult(
                preprocessor=PreprocessorSpec(
                    type=PreprocessorType.RETRIEVAL,
                    inject_as="context",
                    description="file lookup",
                ),
                success=True,
                output="file contents here",
            ),
        ]
        new_messages = PreprocessorPipeline.inject(messages, results)
        assert len(new_messages) == 2
        assert "[context:file lookup]" in new_messages[0]["content"]

    def test_inject_skips_failed_results(self):
        messages = [{"role": "user", "content": "Hello"}]
        results = [
            PreprocessorResult(
                preprocessor=PreprocessorSpec(type=PreprocessorType.COMMAND),
                success=False,
                error="something failed",
            ),
        ]
        new_messages = PreprocessorPipeline.inject(messages, results)
        assert new_messages == messages

    def test_inject_skips_empty_output(self):
        messages = [{"role": "user", "content": "Hello"}]
        results = [
            PreprocessorResult(
                preprocessor=PreprocessorSpec(type=PreprocessorType.COMMAND),
                success=True,
                output="",
            ),
        ]
        new_messages = PreprocessorPipeline.inject(messages, results)
        assert new_messages == messages

    def test_inject_not_mutates_original(self):
        messages = [{"role": "user", "content": "Hello"}]
        original_len = len(messages)
        results = [
            PreprocessorResult(
                preprocessor=PreprocessorSpec(
                    type=PreprocessorType.COMMAND,
                    description="test",
                ),
                success=True,
                output="data",
            ),
        ]
        new_messages = PreprocessorPipeline.inject(messages, results)
        assert len(messages) == original_len
        assert len(new_messages) == original_len + 1

    def test_inject_multiple_results(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Hello"},
        ]
        results = [
            PreprocessorResult(
                preprocessor=PreprocessorSpec(type=PreprocessorType.COMMAND, description="r1"),
                success=True,
                output="data1",
            ),
            PreprocessorResult(
                preprocessor=PreprocessorSpec(type=PreprocessorType.COMMAND, description="r2"),
                success=True,
                output="data2",
            ),
        ]
        new_messages = PreprocessorPipeline.inject(messages, results)
        assert len(new_messages) == 4
        # Both injections should be between system and user
        assert new_messages[0]["role"] == "system"
        assert "[r1]" in new_messages[1]["content"]
        assert "[r2]" in new_messages[2]["content"]
        assert new_messages[3]["role"] == "user"
