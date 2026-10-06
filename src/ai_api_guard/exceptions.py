"""Errors raised by AI API Guard."""

import math


class AIGuardError(Exception):
    """Base error for failures raised by AI API Guard."""


class ProviderError(AIGuardError):
    """Base error for failures that originate at an AI provider.

    ``retry_after`` is an optional server-requested delay, in seconds.
    :class:`~ai_api_guard.guard.AIGuard` uses it only for retryable errors.
    """

    retry_after: float | None

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        """Store the message and an optional server retry delay.

        Args:
            message: Error message. Defaults to an empty string.
            retry_after: Seconds the provider asked the caller to wait, or
                ``None`` when no delay was supplied. A supplied value must be
                finite and >= 0.

        Raises:
            TypeError: ``retry_after`` is not a number or ``None``.
            ValueError: ``retry_after`` is negative or non-finite.
        """
        super().__init__(message)
        self.retry_after = _validate_retry_after(retry_after)


class AuthenticationError(ProviderError):
    """The provider rejected the caller's credentials."""


class RateLimitError(ProviderError):
    """The provider refused the call because a rate limit was exceeded."""


class ProviderTimeoutError(ProviderError):
    """The provider did not answer before the call timed out."""


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached or is temporarily unavailable."""


def _validate_retry_after(value: object) -> float | None:
    """Return a finite delay in seconds, or None when no delay was supplied."""
    if value is None:
        return None
    if isinstance(value, bool):
        message = "retry_after must be a non-negative number or None, got bool"
        raise TypeError(message)
    if isinstance(value, int):
        if value < 0:
            message = f"retry_after must be >= 0, got {value}"
            raise ValueError(message)
        return float(value)
    if isinstance(value, float):
        if math.isnan(value):
            message = "retry_after must not be NaN"
            raise ValueError(message)
        if math.isinf(value):
            sign = "positive" if value > 0 else "negative"
            message = f"retry_after must not be {sign} infinity"
            raise ValueError(message)
        if value < 0:
            message = f"retry_after must be >= 0, got {value}"
            raise ValueError(message)
        return value
    message = (
        "retry_after must be a non-negative number or None, "
        f"got {type(value).__name__}"
    )
    raise TypeError(message)
