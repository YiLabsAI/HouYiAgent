"""Tests for neuro-symbolic engine."""

import pytest

from houyi.verification.config import VerificationConfig
from houyi.verification.neuro_symbolic_engine import NeuroSymbolicEngine, VerificationMetrics
from houyi.verification.sql_verifier import SQLVerifier
from houyi.verification.verifier import VerificationRule


@pytest.fixture
def engine():
    """Create neuro-symbolic engine instance."""
    return NeuroSymbolicEngine()


@pytest.fixture
def sql_verifier():
    """Create SQL verifier instance."""
    return SQLVerifier()


@pytest.fixture
def sql_rule():
    """Create SQL verification rule."""
    return VerificationRule(
        rule_id="test_sql",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": True},
    )


@pytest.mark.asyncio
async def test_generate_and_verify_success(engine, sql_verifier, sql_rule):
    """Test successful generation and verification."""

    async def generator(feedback_context):
        return "SELECT * FROM users;"

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    assert success is True
    assert output == "SELECT * FROM users;"


@pytest.mark.asyncio
async def test_generate_and_verify_auto_fix(engine, sql_verifier, sql_rule):
    """Test auto-fix during verification."""
    call_count = 0

    async def generator(feedback_context):
        nonlocal call_count
        call_count += 1
        # First call returns SQL without semicolon
        return "SELECT * FROM users"

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    # Should auto-fix and succeed
    assert success is True
    assert ";" in output


@pytest.mark.asyncio
async def test_generate_and_verify_disabled(sql_verifier, sql_rule):
    """Test verification disabled mode."""
    config = VerificationConfig.disabled()
    engine = NeuroSymbolicEngine(config=config)

    async def generator(feedback_context):
        return "INVALID SQL"

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    assert success is True
    assert output == "INVALID SQL"


@pytest.mark.asyncio
async def test_generate_and_verify_audit_mode(sql_verifier, sql_rule):
    """Test audit mode - log but don't block."""
    config = VerificationConfig.audit()
    engine = NeuroSymbolicEngine(config=config)

    async def generator(feedback_context):
        return "SELECT * FROM users"  # Missing semicolon

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    # Audit mode doesn't block
    assert success is True


@pytest.mark.asyncio
async def test_generate_and_verify_strict_mode(sql_verifier, sql_rule):
    """Test strict mode - fail immediately."""
    config = VerificationConfig.strict()
    engine = NeuroSymbolicEngine(config=config)

    async def generator(feedback_context):
        return "SELECT * FROM users"  # Missing semicolon

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    # Strict mode fails immediately
    assert success is False


def test_metrics_recording(engine):
    """Test metrics are recorded correctly."""
    engine.metrics.record_verification(True)
    engine.metrics.record_verification(False)
    engine.metrics.record_auto_fix()
    engine.metrics.record_escalation()
    engine.metrics.record_retry()

    stats = engine.get_metrics()

    assert stats["total"] == 2
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["auto_fixed"] == 1
    assert stats["escalated"] == 1
    assert stats["retries"] == 1
    assert stats["success_rate"] == 0.5


@pytest.mark.asyncio
async def test_escalation_with_approval(sql_verifier, sql_rule):
    """Test escalation with human approval."""
    import asyncio

    from houyi.verification.review_queue import ReviewQueue

    config = VerificationConfig.lenient()
    review_queue = ReviewQueue()
    engine = NeuroSymbolicEngine(config=config, review_queue=review_queue)

    async def generator(feedback_context):
        return "SELECT * FROM users WHERE id = 1 OR 1=1;"  # Security error

    async def simulate_human_approval():
        """Simulate human reviewer approving the request."""
        await asyncio.sleep(0.1)
        pending = review_queue.get_pending_requests()
        if pending:
            await review_queue.approve(pending[0].request_id, "test_reviewer")

    # Run verification and human approval concurrently
    verify_task = asyncio.create_task(
        engine.generate_and_verify(generator, sql_verifier, sql_rule, task_id="approved_sql")
    )
    approval_task = asyncio.create_task(simulate_human_approval())

    output, success = await verify_task
    await approval_task

    # Should succeed after human approval
    assert success is True

    # Check metrics
    stats = engine.get_metrics()
    assert stats["escalated"] >= 1


@pytest.mark.asyncio
async def test_metrics_integration(sql_verifier, sql_rule):
    """Test metrics are properly recorded during verification."""
    import asyncio

    from houyi.verification.review_queue import ReviewQueue

    config = VerificationConfig.lenient()
    review_queue = ReviewQueue()
    engine = NeuroSymbolicEngine(config=config, review_queue=review_queue)

    # Successful verification
    async def good_generator(feedback_context):
        return "SELECT * FROM users;"

    output1, success1 = await engine.generate_and_verify(good_generator, sql_verifier, sql_rule)
    assert success1 is True

    # Failed verification with escalation - simulate human rejection
    async def bad_generator(feedback_context):
        return "SELECT * FROM users WHERE id = 1 OR 1=1;"  # Security error

    async def simulate_human_review():
        """Simulate human reviewer rejecting the request."""
        await asyncio.sleep(0.1)
        pending = review_queue.get_pending_requests()
        if pending:
            await review_queue.reject(pending[0].request_id, "test_reviewer", "Security violation")

    # Run verification and human review concurrently
    verify_task = asyncio.create_task(
        engine.generate_and_verify(bad_generator, sql_verifier, sql_rule, task_id="bad_sql")
    )
    review_task = asyncio.create_task(simulate_human_review())

    output2, success2 = await verify_task
    await review_task

    assert success2 is False

    # Check metrics
    stats = engine.get_metrics()
    assert stats["total"] == 2
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["escalated"] >= 1


@pytest.mark.asyncio
async def test_generate_and_verify_task_id(engine, sql_verifier, sql_rule):
    """Test with custom task_id."""

    async def generator(feedback_context):
        return "SELECT * FROM users;"

    output, success = await engine.generate_and_verify(
        generator, sql_verifier, sql_rule, task_id="custom_task_123"
    )

    assert success is True


@pytest.mark.asyncio
async def test_lenient_mode_with_auto_fix(sql_verifier, sql_rule):
    """Test lenient mode enables auto-fix."""
    config = VerificationConfig.lenient()
    engine = NeuroSymbolicEngine(config=config)

    async def generator(feedback_context):
        return "SELECT * FROM users"  # Missing semicolon

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    assert success is True
    assert output.endswith(";")

    # Check metrics
    stats = engine.get_metrics()
    assert stats["auto_fixed"] == 1


def test_metrics_zero_division():
    """Test metrics with zero verifications."""
    metrics = VerificationMetrics()
    stats = metrics.get_stats()

    assert stats["total"] == 0
    assert stats["success_rate"] == 0.0


def test_metrics_all_passed():
    """Test metrics with all passed."""
    metrics = VerificationMetrics()
    metrics.record_verification(True)
    metrics.record_verification(True)

    stats = metrics.get_stats()
    assert stats["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_generate_and_verify_generator_exception(engine, sql_verifier, sql_rule):
    """Test handling generator exceptions."""

    async def generator():
        raise ValueError("Generator failed")

    output, success = await engine.generate_and_verify(generator, sql_verifier, sql_rule)

    # Should handle exception gracefully
    assert success is False
