"""Tests for the provider-independent chat contract."""

import pytest

from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider


class _EchoProvider(AIProvider):
    """In-test provider that echoes the last message."""

    @property
    def name(self) -> str:
        return "echo"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        content = messages[-1]["content"] if messages else ""
        return AIResponse(content=content, model=model)


def test_ai_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AIProvider()  # type: ignore[abstract]


def test_concrete_provider_implements_ai_provider() -> None:
    provider = _EchoProvider()

    assert isinstance(provider, AIProvider)


def test_concrete_provider_returns_ai_response() -> None:
    provider = _EchoProvider()

    response = provider.chat(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert isinstance(response, AIResponse)
    assert response.content == "hello"
    assert response.model == "test-model"


def test_provider_name() -> None:
    provider = _EchoProvider()

    assert provider.name == "echo"
