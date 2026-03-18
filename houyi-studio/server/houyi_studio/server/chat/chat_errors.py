from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ChatErrorCode = str


@dataclass(slots=True)
class ChatErrorInfo:
    error_code: ChatErrorCode
    public_message: str
    retryable: bool
    status_code: int | None = None
    provider_code: str | None = None
    category: str = "unknown"
    internal_message: str | None = None

    def to_transport_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["error"] = self.public_message
        return payload


_RATE_LIMIT_CODES = {"RESOURCE_EXHAUSTED", "RATE_LIMIT", "RATE_LIMITED", "TOO_MANY_REQUESTS"}
_AUTH_CODES = {"UNAUTHENTICATED", "INVALID_AUTH", "INVALID_API_KEY"}
_PERMISSION_CODES = {"PERMISSION_DENIED", "FORBIDDEN"}
_BILLING_CODES = {"INSUFFICIENT_BALANCE", "INSUFFICIENT_CREDITS", "QUOTA_EXHAUSTED", "BILLING"}
_TIMEOUT_CODES = {"DEADLINE_EXCEEDED", "ETIMEDOUT", "ESOCKETTIMEDOUT", "ECONNABORTED"}
_NETWORK_CODES = {"ECONNRESET", "EPIPE", "ENOTFOUND", "ECONNREFUSED"}


def _get_error_message(error: Exception | str | Any) -> str:
    if isinstance(error, Exception):
        return str(error).strip()
    return str(error).strip()


def _get_status_code(error: Exception | Any) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _get_provider_code(error: Exception | Any) -> str | None:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.strip():
        return code.strip().upper()
    return None


def _contains_any(message_upper: str, signals: set[str]) -> bool:
    return any(signal in message_upper for signal in signals)


def normalize_chat_error(error: Exception | str | Any) -> ChatErrorInfo:
    message = _get_error_message(error)
    message_upper = message.upper()
    status_code = _get_status_code(error)
    provider_code = _get_provider_code(error)

    if (
        status_code == 429
        or provider_code in _RATE_LIMIT_CODES
        or "429" in message
        or _contains_any(message_upper, _RATE_LIMIT_CODES)
    ):
        return ChatErrorInfo(
            error_code="provider_rate_limited",
            category="rate_limit",
            status_code=429,
            provider_code=provider_code
            or ("RESOURCE_EXHAUSTED" if "RESOURCE_EXHAUSTED" in message_upper else None),
            retryable=True,
            public_message="The model is temporarily rate limited. Please retry in a moment.",
            internal_message=message or None,
        )

    if (
        status_code == 401
        or provider_code in _AUTH_CODES
        or "401" in message
        or _contains_any(message_upper, _AUTH_CODES)
    ):
        return ChatErrorInfo(
            error_code="provider_auth_failed",
            category="auth",
            status_code=401,
            provider_code=provider_code
            or ("UNAUTHENTICATED" if "UNAUTHENTICATED" in message_upper else None),
            retryable=False,
            public_message="The request failed due to authentication issues. Check the configured credentials before retrying.",
            internal_message=message or None,
        )

    if (
        provider_code in _BILLING_CODES
        or "INSUFFICIENT BALANCE" in message_upper
        or "INSUFFICIENT CREDITS" in message_upper
        or "ACCOUNT BALANCE IS INSUFFICIENT" in message_upper
        or "QUOTA EXHAUSTED" in message_upper
        or '"CODE":30001' in message_upper
    ):
        return ChatErrorInfo(
            error_code="provider_quota_exhausted",
            category="billing",
            status_code=402 if status_code is None else status_code,
            provider_code=provider_code
            or (
                "INSUFFICIENT_BALANCE"
                if "INSUFFICIENT BALANCE" in message_upper
                or "ACCOUNT BALANCE IS INSUFFICIENT" in message_upper
                or '"CODE":30001' in message_upper
                else "QUOTA_EXHAUSTED"
            ),
            retryable=False,
            public_message="The configured model provider has insufficient balance or credits. Check the provider billing status or switch to another API key.",
            internal_message=message or None,
        )

    if (
        status_code == 403
        or provider_code in _PERMISSION_CODES
        or "403" in message
        or _contains_any(message_upper, _PERMISSION_CODES)
    ):
        return ChatErrorInfo(
            error_code="provider_permission_denied",
            category="permission",
            status_code=403,
            provider_code=provider_code
            or (
                "PERMISSION_DENIED"
                if "PERMISSION_DENIED" in message_upper
                else "FORBIDDEN"
                if "FORBIDDEN" in message_upper
                else None
            ),
            retryable=False,
            public_message="The request failed due to missing permissions. Check the configured credentials and project access before retrying.",
            internal_message=message or None,
        )

    if (
        status_code == 408
        or provider_code in _TIMEOUT_CODES
        or "TIMEOUT" in message_upper
        or "TIMED OUT" in message_upper
        or "DEADLINE_EXCEEDED" in message_upper
    ):
        return ChatErrorInfo(
            error_code="provider_timeout",
            category="timeout",
            status_code=408,
            provider_code=provider_code
            or ("DEADLINE_EXCEEDED" if "DEADLINE_EXCEEDED" in message_upper else None),
            retryable=True,
            public_message="The request timed out before the model finished responding. Please retry or reduce the request size.",
            internal_message=message or None,
        )

    if provider_code in _NETWORK_CODES or any(
        signal in message_upper
        for signal in ("UNEXPECTED EOF", "ECONNRESET", "NETWORK ERROR", "CONNECTION RESET")
    ):
        return ChatErrorInfo(
            error_code="provider_network_error",
            category="network",
            status_code=503,
            provider_code=provider_code
            or ("ECONNRESET" if "ECONNRESET" in message_upper else None),
            retryable=True,
            public_message="The connection to the model was interrupted. Please retry in a moment.",
            internal_message=message or None,
        )

    return ChatErrorInfo(
        error_code="provider_request_failed",
        category="unknown",
        status_code=status_code,
        provider_code=provider_code,
        retryable=True,
        public_message="The model request failed. Please retry in a moment.",
        internal_message=message or None,
    )


def build_stream_error_content(error: Exception | str | Any) -> str:
    info = normalize_chat_error(error)
    if info.error_code == "provider_rate_limited":
        return "The model is temporarily rate limited by the provider. Please retry in a moment."
    if info.error_code == "provider_auth_failed":
        return "The model request could not be authenticated. Check the configured credentials and retry."
    if info.error_code == "provider_permission_denied":
        return "The model request was blocked due to missing permissions. Check the configured credentials and project access before retrying."
    if info.error_code == "provider_quota_exhausted":
        return "The configured model provider account has insufficient balance or credits. Check the provider billing status or switch to another API key."
    if info.error_code == "provider_timeout":
        return "The model request timed out before the response completed. Please retry or reduce the request size."
    if info.error_code == "provider_network_error":
        return "The connection to the model was interrupted. Please retry in a moment."
    return info.public_message


def build_public_stream_error_message(error: Exception | str | Any) -> str:
    return normalize_chat_error(error).public_message


def build_transport_chat_error(error: Exception | str | Any) -> dict[str, Any]:
    return normalize_chat_error(error).to_transport_dict()
