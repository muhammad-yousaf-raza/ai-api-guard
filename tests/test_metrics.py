"""Tests for request metrics and AIGuard observability."""

import pytest

from ai_api_guard import AIGuard, RequestMetrics, RetryPolicy
from ai_api_guard.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider


class _Clock:
    """Returns scripted timestamps."""

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        if not self._times:
            message = "clock has no remaining timestamps"
            raise AssertionError(message)
        return self._times.pop(0)


class _MetricsSink:
    """Records callback arguments and can fail on purpose."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.records: list[RequestMetrics] = []
        self._error = error

    def __call__(self, metrics: RequestMetrics) -> None:
        self.records.append(metrics)
        if self._error is not None:
            raise self._error


class _OutcomeProvider(AIProvider):
    """In-test provider with a fixed name and scripted outcomes."""

    def __init__(self, name: str, outcomes: list[AIResponse | Exception]) -> None:
        self._name = name
        self._outcomes = list(outcomes)
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        del model, messages
        self.calls += 1
        if not self._outcomes:
            message = "provider has no remaining outcomes"
            raise AssertionError(message)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _guard(
    provider: AIProvider,
    *,
    clock: _Clock,
    sink: _MetricsSink,
    max_attempts: int = 3,
) -> AIGuard:
    return AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_delay=0.0),
        sleep=lambda _delay: None,
        on_metrics=sink,
        clock=clock,
    )


def test_valid_success_metrics() -> None:
    metrics = RequestMetrics(
        provider="echo",
        model="test-model",
        attempts=1,
        latency=0.25,
        success=True,
    )

    assert metrics.provider == "echo"
    assert metrics.model == "test-model"
    assert metrics.attempts == 1
    assert metrics.latency == 0.25
    assert metrics.success is True
    assert metrics.error_type is None


def test_valid_failure_metrics() -> None:
    metrics = RequestMetrics(
        provider="echo",
        model="test-model",
        attempts=3,
        latency=1.5,
        success=False,
        error_type="RateLimitError",
    )

    assert metrics.success is False
    assert metrics.error_type == "RateLimitError"


def test_empty_provider_rejected() -> None:
    with pytest.raises(ValueError, match="provider"):
        RequestMetrics(
            provider="",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=True,
        )


def test_empty_model_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        RequestMetrics(
            provider="echo",
            model="",
            attempts=1,
            latency=0.0,
            success=True,
        )


def test_attempts_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="attempts"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=0,
            latency=0.0,
            success=True,
        )


def test_negative_latency_rejected() -> None:
    with pytest.raises(ValueError, match="latency"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=-0.1,
            success=True,
        )


def test_success_with_error_type_rejected() -> None:
    with pytest.raises(ValueError, match="error_type to be None"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=True,
            error_type="RateLimitError",
        )


def test_failure_without_error_type_rejected() -> None:
    with pytest.raises(ValueError, match="requires error_type"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=False,
        )


def test_failure_with_empty_error_type_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=False,
            error_type="",
        )


def test_successful_first_attempt_emits_one_metrics_record() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _OutcomeProvider("custom-provider", [response])
    sink = _MetricsSink()
    guard = _guard(provider, clock=_Clock([10.0, 10.5]), sink=sink)

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.provider == "custom-provider"
    assert metrics.model == "test-model"
    assert metrics.attempts == 1
    assert metrics.latency == 0.5
    assert metrics.success is True
    assert metrics.error_type is None
    assert provider.calls == 1


def test_retry_then_success_emits_one_metrics_record() -> None:
    response = AIResponse(content="recovered", model="test-model")
    provider = _OutcomeProvider(
        "custom-provider",
        [RateLimitError("again"), ProviderUnavailableError("later"), response],
    )
    sink = _MetricsSink()
    guard = _guard(provider, clock=_Clock([1.0, 2.5]), sink=sink)

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.attempts == 3
    assert metrics.success is True
    assert metrics.error_type is None
    assert metrics.latency == 1.5
    assert provider.calls == 3


def test_final_retry_failure_emits_one_failure_record() -> None:
    final_error = RateLimitError("final")
    provider = _OutcomeProvider(
        "custom-provider",
        [RateLimitError("first"), ProviderUnavailableError("second"), final_error],
    )
    sink = _MetricsSink()
    guard = _guard(provider, clock=_Clock([4.0, 4.2]), sink=sink)

    with pytest.raises(RateLimitError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is final_error
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.attempts == 3
    assert metrics.success is False
    assert metrics.error_type == "RateLimitError"
    assert provider.calls == 3


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("denied"),
        ProviderError("generic"),
        RuntimeError("boom"),
    ],
)
def test_non_retryable_failure_emits_metrics(error: Exception) -> None:
    provider = _OutcomeProvider("custom-provider", [error])
    sink = _MetricsSink()
    guard = _guard(provider, clock=_Clock([0.0, 0.2]), sink=sink, max_attempts=3)

    with pytest.raises(type(error)) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error
    assert provider.calls == 1
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.success is False
    assert metrics.attempts == 1
    assert metrics.error_type == type(error).__name__


def test_metrics_callback_exception_does_not_break_success() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _OutcomeProvider("custom-provider", [response])
    sink = _MetricsSink(error=RuntimeError("observer failed"))
    guard = _guard(provider, clock=_Clock([1.0, 1.1]), sink=sink)

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert len(sink.records) == 1


def test_metrics_callback_exception_preserves_provider_failure() -> None:
    error = AuthenticationError("denied")
    provider = _OutcomeProvider("custom-provider", [error])
    sink = _MetricsSink(error=RuntimeError("observer failed"))
    guard = _guard(provider, clock=_Clock([1.0, 1.1]), sink=sink)

    with pytest.raises(AuthenticationError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error
    assert len(sink.records) == 1
    assert sink.records[0].error_type == "AuthenticationError"


def test_negative_clock_difference_is_clamped_to_zero() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _OutcomeProvider("custom-provider", [response])
    sink = _MetricsSink()
    guard = _guard(provider, clock=_Clock([5.0, 4.0]), sink=sink)

    guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert sink.records[0].latency == 0.0
