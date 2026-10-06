"""Tests for AIGuard request delegation and retries."""

import pytest

from ai_api_guard import AIGuard, RetryPolicy
from ai_api_guard.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider


class _RecordingProvider(AIProvider):
    """In-test provider that records the arguments it receives."""

    def __init__(
        self,
        *,
        response: AIResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.received_model: str | None = None
        self.received_messages: list[dict[str, str]] | None = None
        self._response = response
        self._error = error

    @property
    def name(self) -> str:
        return "recording"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        self.received_model = model
        self.received_messages = messages
        if self._error is not None:
            raise self._error
        if self._response is None:
            message = "recording provider has no response"
            raise AssertionError(message)
        return self._response


def test_guard_exposes_supplied_provider() -> None:
    provider = _RecordingProvider()
    guard = AIGuard(provider)

    assert guard.provider is provider


def test_chat_delegates_model_and_messages_unchanged() -> None:
    response = AIResponse(content="hello", model="test-model")
    provider = _RecordingProvider(response=response)
    guard = AIGuard(provider)
    messages = [{"role": "user", "content": "Hello"}]

    result = guard.chat(model="test-model", messages=messages)

    assert provider.received_model == "test-model"
    assert provider.received_messages is messages
    assert result is response


def test_provider_exceptions_propagate_unchanged() -> None:
    error = RateLimitError("limited")
    provider = _RecordingProvider(error=error)
    guard = AIGuard(provider, sleep=_SleepRecorder())

    with pytest.raises(RateLimitError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error


class _SleepRecorder:
    """Records delays without waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class _ScriptedProvider(AIProvider):
    """In-test provider that returns or raises a scripted sequence."""

    def __init__(self, outcomes: list[AIResponse | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.models: list[str] = []
        self.message_lists: list[list[dict[str, str]]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        self.models.append(model)
        self.message_lists.append(messages)
        if not self._outcomes:
            message = "scripted provider has no remaining outcomes"
            raise AssertionError(message)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_success_on_first_call_does_not_sleep() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider([response])
    sleep = _SleepRecorder()
    messages = [{"role": "user", "content": "Hello"}]
    guard = AIGuard(provider, sleep=sleep)

    result = guard.chat(model="test-model", messages=messages)

    assert result is response
    assert sleep.delays == []
    assert provider.models == ["test-model"]
    assert provider.message_lists == [messages]


@pytest.mark.parametrize(
    "error_type",
    [RateLimitError, ProviderTimeoutError, ProviderUnavailableError],
)
def test_transient_provider_errors_are_retried(
    error_type: type[RateLimitError],
) -> None:
    error = error_type("transient")
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider([error, response])
    sleep = _SleepRecorder()
    policy = RetryPolicy(max_attempts=3, initial_delay=0.5, backoff_multiplier=2.0)
    guard = AIGuard(provider, retry_policy=policy, sleep=sleep)
    messages = [{"role": "user", "content": "Hello"}]

    result = guard.chat(model="test-model", messages=messages)

    assert result is response
    assert provider.models == ["test-model", "test-model"]
    assert provider.message_lists == [messages, messages]
    assert sleep.delays == [0.5]


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("denied"),
        ProviderError("generic"),
        RuntimeError("boom"),
    ],
)
def test_non_retryable_exceptions_are_not_retried(error: Exception) -> None:
    provider = _ScriptedProvider([error])
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.5),
        sleep=sleep,
    )

    with pytest.raises(type(error)) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error
    assert sleep.delays == []
    assert provider.models == ["test-model"]


def test_retry_stops_after_max_attempts() -> None:
    errors = [RateLimitError("one"), RateLimitError("two"), RateLimitError("three")]
    provider = _ScriptedProvider(list(errors))
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.5),
        sleep=sleep,
    )

    with pytest.raises(RateLimitError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is errors[2]
    assert len(provider.models) == 3
    assert len(sleep.delays) == 2


def test_sleep_receives_exponential_delays() -> None:
    errors = [ProviderUnavailableError(str(index)) for index in range(6)]
    provider = _ScriptedProvider(list(errors))
    sleep = _SleepRecorder()
    policy = RetryPolicy(
        max_attempts=6,
        initial_delay=0.5,
        max_delay=8.0,
        backoff_multiplier=2.0,
    )
    guard = AIGuard(provider, retry_policy=policy, sleep=sleep)

    with pytest.raises(ProviderUnavailableError):
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert sleep.delays == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_success_after_retries_returns_same_response() -> None:
    response = AIResponse(content="recovered", model="test-model")
    provider = _ScriptedProvider(
        [
            RateLimitError("one"),
            ProviderUnavailableError("two"),
            response,
        ],
    )
    sleep = _SleepRecorder()
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        backoff_multiplier=2.0,
    )
    guard = AIGuard(provider, retry_policy=policy, sleep=sleep)

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [0.5, 1.0]


def test_final_retryable_exception_is_same_instance() -> None:
    final_error = ProviderTimeoutError("final")
    provider = _ScriptedProvider(
        [
            RateLimitError("first"),
            ProviderUnavailableError("second"),
            final_error,
        ],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.25, backoff_multiplier=2.0),
        sleep=sleep,
    )

    with pytest.raises(ProviderTimeoutError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is final_error
    assert sleep.delays == [0.25, 0.5]


def test_retry_after_overrides_exponential_delay() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider(
        [RateLimitError("limited", retry_after=3.0), response],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.5),
        sleep=sleep,
    )

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [3.0]


def test_retry_after_is_capped_by_max_delay() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider(
        [ProviderTimeoutError("slow", retry_after=30.0), response],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.5, max_delay=8.0),
        sleep=sleep,
    )

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [8.0]


def test_zero_retry_after_sleeps_zero_seconds() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider(
        [ProviderUnavailableError("down", retry_after=0.0), response],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.5),
        sleep=sleep,
    )

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [0.0]


def test_later_attempt_without_retry_after_uses_exponential_delay() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider(
        [
            RateLimitError("first", retry_after=3.0),
            ProviderUnavailableError("second"),
            response,
        ],
    )
    sleep = _SleepRecorder()
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=8.0,
        backoff_multiplier=2.0,
    )
    guard = AIGuard(provider, retry_policy=policy, sleep=sleep)

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [3.0, 1.0]


def test_retry_after_on_non_retryable_error_is_ignored() -> None:
    error = AuthenticationError("denied", retry_after=5.0)
    provider = _ScriptedProvider([error])
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay=0.5),
        sleep=sleep,
    )

    with pytest.raises(AuthenticationError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error
    assert sleep.delays == []


def test_final_exception_identity_is_preserved_when_retry_after_is_set() -> None:
    final_error = ProviderTimeoutError("final", retry_after=4.0)
    provider = _ScriptedProvider(
        [
            RateLimitError("first", retry_after=1.5),
            final_error,
        ],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.5),
        sleep=sleep,
    )

    with pytest.raises(ProviderTimeoutError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is final_error
    assert sleep.delays == [1.5]


def _forbidden_random() -> float:
    message = "random source must not be called"
    raise AssertionError(message)


def test_sleep_uses_jittered_exponential_delay() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider([RateLimitError("limited"), response])
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_delay=0.5,
            jitter=1.0,
            unit_random=lambda: 0.0,
        ),
        sleep=sleep,
    )

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [0.25]


def test_retry_after_stays_exact_when_jitter_is_enabled() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider(
        [RateLimitError("limited", retry_after=3.0), response],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=2,
            jitter=1.0,
            unit_random=_forbidden_random,
        ),
        sleep=sleep,
    )

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [3.0]


def test_retry_after_then_jittered_exponential_delay() -> None:
    response = AIResponse(content="ok", model="test-model")
    provider = _ScriptedProvider(
        [
            RateLimitError("first", retry_after=3.0),
            ProviderUnavailableError("second"),
            response,
        ],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=8.0,
            backoff_multiplier=2.0,
            jitter=1.0,
            unit_random=lambda: 0.0,
        ),
        sleep=sleep,
    )

    result = guard.chat(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is response
    assert sleep.delays == [3.0, 0.5]


def test_jitter_does_not_retry_non_retryable_errors() -> None:
    error = AuthenticationError("denied", retry_after=5.0)
    provider = _ScriptedProvider([error])
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=3,
            jitter=1.0,
            unit_random=_forbidden_random,
        ),
        sleep=sleep,
    )

    with pytest.raises(AuthenticationError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error
    assert sleep.delays == []


def test_final_exception_identity_is_preserved_with_jitter() -> None:
    final_error = ProviderTimeoutError("final")
    provider = _ScriptedProvider(
        [RateLimitError("first"), final_error],
    )
    sleep = _SleepRecorder()
    guard = AIGuard(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_delay=0.5,
            jitter=1.0,
            unit_random=lambda: 0.0,
        ),
        sleep=sleep,
    )

    with pytest.raises(ProviderTimeoutError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is final_error
    assert sleep.delays == [0.25]


def test_ai_guard_imports_from_package() -> None:
    import ai_api_guard

    assert ai_api_guard.AIGuard is AIGuard
