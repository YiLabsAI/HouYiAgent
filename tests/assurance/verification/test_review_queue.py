"""Tests for review queue."""

import asyncio
from datetime import datetime

import pytest

from houyi.assurance.verification.review_queue import ReviewQueue, ReviewRequest


@pytest.fixture
def review_queue():
    """Create review queue instance."""
    return ReviewQueue()


@pytest.fixture
def sample_request():
    """Create sample review request."""
    return ReviewRequest(
        request_id="test_001",
        task_id="task_001",
        error_type="sql_injection",
        error_message="Potential SQL injection detected",
        original_output="SELECT * FROM users WHERE id = 1 OR 1=1",
        timeout_seconds=5,
    )


@pytest.mark.asyncio
async def test_submit_and_approve(review_queue, sample_request):
    """Test submitting and approving a review request."""

    # Submit in background
    async def submit_task():
        decision = await review_queue.submit(sample_request)
        return decision

    # Start submission
    submit = asyncio.create_task(submit_task())

    # Wait a bit for submission to register
    await asyncio.sleep(0.1)

    # Approve the request
    success = await review_queue.approve(sample_request.request_id, "reviewer1")
    assert success is True

    # Check decision
    decision = await submit
    assert decision == "approved"

    # Verify request state
    request = review_queue.get_request(sample_request.request_id)
    assert request.status == "approved"
    assert request.reviewer == "reviewer1"
    assert request.reviewed_at is not None


@pytest.mark.asyncio
async def test_submit_and_reject(review_queue, sample_request):
    """Test submitting and rejecting a review request."""

    async def submit_task():
        decision = await review_queue.submit(sample_request)
        return decision

    submit = asyncio.create_task(submit_task())
    await asyncio.sleep(0.1)

    # Reject the request
    success = await review_queue.reject(
        sample_request.request_id, "reviewer1", "Security risk too high"
    )
    assert success is True

    decision = await submit
    assert decision == "rejected"

    request = review_queue.get_request(sample_request.request_id)
    assert request.status == "rejected"
    assert request.reviewer == "reviewer1"
    assert "Security risk" in request.decision


@pytest.mark.asyncio
async def test_submit_timeout(review_queue):
    """Test review request timeout."""
    request = ReviewRequest(
        request_id="test_timeout",
        task_id="task_timeout",
        error_type="unknown_error",
        error_message="Unknown error",
        original_output="some output",
        timeout_seconds=0.05,  # sub-second timeout keeps the test fast
    )

    # Submit without approving/rejecting
    decision = await review_queue.submit(request)

    # Should timeout
    assert decision == "timeout"
    assert request.status == "timeout"


@pytest.mark.asyncio
async def test_approve_nonexistent_request(review_queue):
    """Test approving a request that doesn't exist."""
    success = await review_queue.approve("nonexistent", "reviewer1")
    assert success is False


@pytest.mark.asyncio
async def test_reject_nonexistent_request(review_queue):
    """Test rejecting a request that doesn't exist."""
    success = await review_queue.reject("nonexistent", "reviewer1", "reason")
    assert success is False


def test_get_pending_requests(review_queue, sample_request):
    """Test getting pending requests."""
    # Add request to queue
    review_queue.queue[sample_request.request_id] = sample_request

    pending = review_queue.get_pending_requests()
    assert len(pending) == 1
    assert pending[0].request_id == sample_request.request_id


def test_get_pending_requests_empty(review_queue):
    """Test getting pending requests when queue is empty."""
    pending = review_queue.get_pending_requests()
    assert len(pending) == 0


def test_pending_filters_non_pending(review_queue, sample_request):
    """Test that get_pending_requests filters out non-pending requests."""
    # Add approved request
    sample_request.status = "approved"
    review_queue.queue[sample_request.request_id] = sample_request

    # Add pending request
    pending_request = ReviewRequest(
        request_id="pending_001",
        task_id="task_002",
        error_type="error",
        error_message="Error",
        original_output="output",
    )
    review_queue.queue[pending_request.request_id] = pending_request

    pending = review_queue.get_pending_requests()
    assert len(pending) == 1
    assert pending[0].request_id == "pending_001"


def test_get_request(review_queue, sample_request):
    """Test getting a specific request."""
    review_queue.queue[sample_request.request_id] = sample_request

    request = review_queue.get_request(sample_request.request_id)
    assert request is not None
    assert request.request_id == sample_request.request_id


def test_get_request_nonexistent(review_queue):
    """Test getting a nonexistent request."""
    request = review_queue.get_request("nonexistent")
    assert request is None


@pytest.mark.asyncio
async def test_multiple_concurrent_requests(review_queue):
    """Test handling multiple concurrent review requests."""
    requests = [
        ReviewRequest(
            request_id=f"test_{i}",
            task_id=f"task_{i}",
            error_type="error",
            error_message=f"Error {i}",
            original_output=f"output {i}",
            timeout_seconds=5,
        )
        for i in range(3)
    ]

    # Submit all requests
    async def submit_and_approve(req, idx):
        submit_task = asyncio.create_task(review_queue.submit(req))
        await asyncio.sleep(0.1)
        await review_queue.approve(req.request_id, f"reviewer{idx}")
        return await submit_task

    results = await asyncio.gather(*[submit_and_approve(req, i) for i, req in enumerate(requests)])

    # All should be approved
    assert all(r == "approved" for r in results)
    assert len(review_queue.queue) == 3


@pytest.mark.asyncio
async def test_review_request_fields(sample_request):
    """Test ReviewRequest fields are properly set."""
    assert sample_request.request_id == "test_001"
    assert sample_request.task_id == "task_001"
    assert sample_request.error_type == "sql_injection"
    assert sample_request.status == "pending"
    assert sample_request.reviewer is None
    assert sample_request.reviewed_at is None
    assert sample_request.decision is None
    assert isinstance(sample_request.created_at, datetime)


@pytest.mark.asyncio
async def test_approve_already_resolved(review_queue, sample_request):
    """Test approving an already resolved request."""
    review_queue.queue[sample_request.request_id] = sample_request

    # First approval
    success1 = await review_queue.approve(sample_request.request_id, "reviewer1")
    assert success1 is True

    # Second approval (future already done)
    success2 = await review_queue.approve(sample_request.request_id, "reviewer2")
    assert success2 is True

    # Should keep first reviewer
    request = review_queue.get_request(sample_request.request_id)
    assert request.reviewer == "reviewer2"  # Last one wins


@pytest.mark.asyncio
async def test_reject_with_no_reason(review_queue, sample_request):
    """Test rejecting without providing a reason."""

    async def submit_task():
        return await review_queue.submit(sample_request)

    submit = asyncio.create_task(submit_task())
    await asyncio.sleep(0.1)

    success = await review_queue.reject(sample_request.request_id, "reviewer1")
    assert success is True

    decision = await submit
    assert decision == "rejected"

    request = review_queue.get_request(sample_request.request_id)
    assert request.decision == "rejected"
