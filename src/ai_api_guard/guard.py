"""Public entry point that delegates chat requests to an AI provider."""

import time
from collections.abc import Callable

from ai_api_guard.exceptions import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.metrics import RequestMetrics
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider
from ai_api_guard.retry import RetryEvent, RetryPolicy

_RETRYABLE_ERRORS = (RateLimitError, ProviderTimeoutError, ProviderUnavailableError)


class AIGuard:
    """Main entry point for guarded AI provider requests."""

    def __init__(
        self,
        provider: AIProvider,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_metrics: Callable[[RequestMetrics], None] | None = None,
        on_retry: Callable[[RetryEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store the provider, retry settings, and optional callbacks.

        Args:
            provider: Provider implementation used for every chat call.
            retry_policy: Retry schedule for transient failures. Defaults to
                :class:`~ai_api_guard.retry.RetryPolicy`.
            sleep: Callable used to wait between attempts. Defaults to
                :func:`time.sleep`.
            on_metrics: Optional callback invoked once per chat call with the
                finished request record. Callback failures are ignored.
            on_retry: Optional callback invoked before each wait that will be
                followed by another provider call. Callback failures are
                ignored.
            clock: Monotonic clock used to measure request latency. Defaults
                to :func:`time.monotonic`.
        """
        self._provider = provider
        self._retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self._sleep = sleep
        self._on_metrics = on_metrics
        self._on_retry = on_retry
        self._clock = clock

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
        comes from :meth:`~ai_api_guard.retry.RetryPolicy.delay_for_retry`.
        When the error carries ``retry_after``, that server delay replaces
        the exponential delay for that attempt and is capped at
        ``max_delay``. That server delay is not jittered. When ``jitter``
        is enabled, only the exponential delay is spread. Any other
        exception is raised immediately. A
        successful response is returned unchanged, and the exception from
        the final attempt is re-raised as the same instance.

        ``on_retry`` runs only when another attempt will happen, after the
        wait is chosen and before ``sleep``. The same delay value is slept.
        Callback failures are ignored.

        One :class:`~ai_api_guard.metrics.RequestMetrics` record is emitted
        for the finished call. On success, its token counts and cost are
        copied from the returned response. Failures record zero usage and
        zero cost. Intermediate retry failures are not recorded.

        Args:
            model: Model identifier understood by the provider.
            messages: Ordered chat messages. Each mapping uses string keys and
                string values.

        Returns:
            The response object returned by the provider.
        """
        started = self._clock()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._provider.chat(model=model, messages=messages)
            except _RETRYABLE_ERRORS as error:
                if attempts < self._retry_policy.max_attempts:
                    delay = self._retry_policy.delay_for_retry(
                        attempts,
                        retry_after=error.retry_after,
                    )
                    self._notify_retry(model=model, attempt=attempts, delay=delay, error=error)
                    self._sleep(delay)
                    continue
                self._emit_metrics(
                    model=model,
                    attempts=attempts,
                    started=started,
                    error=error,
                )
                raise
            except Exception as error:
                self._emit_metrics(
                    model=model,
                    attempts=attempts,
                    started=started,
                    error=error,
                )
                raise
            else:
                self._emit_metrics(
                    model=model,
                    attempts=attempts,
                    started=started,
                    error=None,
                    response=response,
                )
                return response

    def _notify_retry(
        self,
        *,
        model: str,
        attempt: int,
        delay: float,
        error: ProviderError,
    ) -> None:
        """Tell ``on_retry`` about a wait that will be followed by another call."""
        if self._on_retry is None:
            return
        event = RetryEvent(
            provider=self._provider.name,
            model=model,
            attempt=attempt,
            next_attempt=attempt + 1,
            delay=delay,
            error_type=type(error).__name__,
            retry_after=error.retry_after,
        )
        try:
            self._on_retry(event)
        except Exception:  # noqa: BLE001
            return

    def _emit_metrics(
        self,
        *,
        model: str,
        attempts: int,
        started: float,
        error: Exception | None,
        response: AIResponse | None = None,
    ) -> None:
        """Record the finished request and notify the metrics callback once."""
        latency = max(0.0, self._clock() - started)
        if error is None and response is not None:
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            cost = response.cost
        else:
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
        metrics = RequestMetrics(
            provider=self._provider.name,
            model=model,
            attempts=attempts,
            latency=latency,
            success=error is None,
            error_type=None if error is None else type(error).__name__,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )
        if self._on_metrics is None:
            return
        try:
            self._on_metrics(metrics)
        except Exception:  # noqa: BLE001
            return
