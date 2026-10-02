"""Public entry point that delegates chat requests to an AI provider."""

import time
from collections.abc import Callable

from ai_api_guard.exceptions import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider
from ai_api_guard.retry import RetryPolicy

_RETRYABLE_ERRORS = (RateLimitError, ProviderTimeoutError, ProviderUnavailableError)


class AIGuard:
    """Main entry point for guarded AI provider requests."""

    def __init__(
        self,
        provider: AIProvider,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Store the provider and retry settings used for chat requests.

        Args:
            provider: Provider implementation used for every chat call.
            retry_policy: Retry schedule for transient failures. Defaults to
                :class:`~ai_api_guard.retry.RetryPolicy`.
            sleep: Callable used to wait between attempts. Defaults to
                :func:`time.sleep`.
        """
        self._provider = provider
        self._retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self._sleep = sleep

    @property
    def provider(self) -> AIProvider:
        """Return the provider this guard delegates to."""
        return self._provider

    @property
    def retry_policy(self) -> RetryPolicy:
        """Return the retry schedule used for transient failures."""
        return self._retry_policy

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        """Send a chat request, retrying transient provider failures.

        ``RateLimitError``, ``ProviderTimeoutError``, and
        ``ProviderUnavailableError`` are retried until ``max_attempts``
        provider calls have been made. The wait before each additional call
        comes from :meth:`~ai_api_guard.retry.RetryPolicy.delay_for_attempt`.
        Any other exception is raised immediately. A successful response is
        returned unchanged, and the exception from the final attempt is
        re-raised as the same instance.

        Args:
            model: Model identifier understood by the provider.
            messages: Ordered chat messages. Each mapping uses string keys and
                string values.

        Returns:
            The response object returned by the provider.
        """
        attempt = 1
        while True:
            try:
                return self._provider.chat(model=model, messages=messages)
            except _RETRYABLE_ERRORS:
                if attempt >= self._retry_policy.max_attempts:
                    raise
                self._sleep(self._retry_policy.delay_for_attempt(attempt))
                attempt += 1
