"""Verification configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class VerificationMode(str, Enum):
    """Verification mode."""

    STRICT = "strict"  # Fail on any error, no auto-fix
    LENIENT = "lenient"  # Auto-fix enabled, retry on failure
    DISABLED = "disabled"  # No verification
    AUDIT = "audit"  # Verify but don't block execution


class EscalationPolicy(BaseModel):
    """Policy for escalating verification failures."""

    notify_channels: list[str] = Field(
        default_factory=list, description="Notification channels (slack, email)"
    )
    timeout_seconds: int = Field(300, description="Timeout for human review")
    default_action: str = Field(
        "fail", description="Default action on timeout: 'fail', 'approve', 'retry'"
    )


class VerificationConfig(BaseModel):
    """Configuration for verification behavior."""

    enabled: bool = Field(True, description="Whether verification is enabled")
    mode: VerificationMode = Field(VerificationMode.LENIENT, description="Verification mode")
    verifiers: list[str] = Field(default_factory=list, description="List of verifiers to use")
    auto_fix: bool = Field(True, description="Whether to attempt auto-fix")
    max_retries: int = Field(3, description="Maximum retry attempts")
    on_failure: str = Field(
        "retry", description="Action on failure: 'retry', 'fail', 'warn', 'escalate'"
    )
    escalation_policy: EscalationPolicy | None = Field(
        None, description="Policy for escalating failures"
    )

    @classmethod
    def strict(cls) -> "VerificationConfig":
        """Create strict verification config."""
        return cls(
            enabled=True,
            mode=VerificationMode.STRICT,
            auto_fix=False,
            max_retries=2,
            on_failure="fail",
        )

    @classmethod
    def lenient(cls) -> "VerificationConfig":
        """Create lenient verification config."""
        return cls(
            enabled=True,
            mode=VerificationMode.LENIENT,
            auto_fix=True,
            max_retries=5,
            on_failure="retry",
        )

    @classmethod
    def disabled(cls) -> "VerificationConfig":
        """Create disabled verification config."""
        return cls(enabled=False)

    @classmethod
    def audit(cls) -> "VerificationConfig":
        """Create audit-only verification config."""
        return cls(
            enabled=True,
            mode=VerificationMode.AUDIT,
            auto_fix=False,
            max_retries=0,
            on_failure="warn",
        )
