"""Human review queue for escalated verification failures."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReviewRequest(BaseModel):
    """Request for human review of verification failure."""

    request_id: str = Field(..., description="Unique request ID")
    task_id: str = Field(..., description="Task that failed verification")
    error_type: str = Field(..., description="Type of verification error")
    error_message: str = Field(..., description="Error message")
    original_output: Any = Field(..., description="Original output that failed")
    suggested_fix: str | None = Field(None, description="Suggested fix")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    timeout_seconds: int = Field(300, description="Timeout for review")
    status: str = Field("pending", description="Status: pending, approved, rejected, timeout")
    reviewer: str | None = Field(None, description="Reviewer who handled this")
    reviewed_at: datetime | None = Field(None, description="Review timestamp")
    decision: str | None = Field(None, description="Review decision")


class ReviewQueue:
    """Queue for managing human review requests."""

    def __init__(self):
        """Initialize review queue."""
        self.queue: dict[str, ReviewRequest] = {}
        self.pending_futures: dict[str, asyncio.Future] = {}

    async def submit(self, request: ReviewRequest) -> str:
        """Submit a request for review.

        Args:
            request: Review request

        Returns:
            Decision: 'approved', 'rejected', or 'timeout'
        """
        self.queue[request.request_id] = request

        # Create future for waiting on decision
        future: asyncio.Future = asyncio.Future()
        self.pending_futures[request.request_id] = future

        logger.info(f"Submitted review request: {request.request_id}")

        # Notify (in real implementation, this would send to Slack/email)
        await self._notify_reviewers(request)

        # Wait for decision with timeout
        try:
            decision = await asyncio.wait_for(future, timeout=request.timeout_seconds)
            return decision
        except asyncio.TimeoutError:
            logger.warning(f"Review request timed out: {request.request_id}")
            request.status = "timeout"
            return "timeout"
        finally:
            # Cleanup
            self.pending_futures.pop(request.request_id, None)

    async def approve(self, request_id: str, reviewer: str) -> bool:
        """Approve a review request.

        Args:
            request_id: Request ID
            reviewer: Reviewer name

        Returns:
            True if approved successfully
        """
        request = self.queue.get(request_id)
        if not request:
            return False

        request.status = "approved"
        request.reviewer = reviewer
        request.reviewed_at = datetime.now()
        request.decision = "approved"

        # Resolve future
        future = self.pending_futures.get(request_id)
        if future and not future.done():
            future.set_result("approved")

        logger.info(f"Review request approved by {reviewer}: {request_id}")
        return True

    async def reject(self, request_id: str, reviewer: str, reason: str | None = None) -> bool:
        """Reject a review request.

        Args:
            request_id: Request ID
            reviewer: Reviewer name
            reason: Rejection reason

        Returns:
            True if rejected successfully
        """
        request = self.queue.get(request_id)
        if not request:
            return False

        request.status = "rejected"
        request.reviewer = reviewer
        request.reviewed_at = datetime.now()
        request.decision = f"rejected: {reason}" if reason else "rejected"

        # Resolve future
        future = self.pending_futures.get(request_id)
        if future and not future.done():
            future.set_result("rejected")

        logger.info(f"Review request rejected by {reviewer}: {request_id}")
        return True

    async def _notify_reviewers(self, request: ReviewRequest) -> None:
        """Notify reviewers of new request.

        Args:
            request: Review request
        """
        # In real implementation, send to Slack/email/dashboard
        logger.info(f"[NOTIFICATION] Review needed: {request.error_type} - {request.error_message}")

    def get_pending_requests(self) -> list[ReviewRequest]:
        """Get all pending review requests.

        Returns:
            List of pending requests
        """
        return [r for r in self.queue.values() if r.status == "pending"]

    def get_request(self, request_id: str) -> ReviewRequest | None:
        """Get a specific review request.

        Args:
            request_id: Request ID

        Returns:
            Review request or None
        """
        return self.queue.get(request_id)
