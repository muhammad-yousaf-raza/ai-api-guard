"""Tests for AIGuard request delegation."""

import pytest

from ai_api_guard import AIGuard
from ai_api_guard.exceptions import RateLimitError
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
    guard = AIGuard(provider)

    with pytest.raises(RateLimitError) as caught:
        guard.chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert caught.value is error


def test_ai_guard_imports_from_package() -> None:
    import ai_api_guard

    assert ai_api_guard.AIGuard is AIGuard
