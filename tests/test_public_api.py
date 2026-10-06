"""Tests for objects exported by the package root."""

import ai_api_guard
from ai_api_guard import (
    AIGuardError,
    AIProvider,
    AIResponse,
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    RetryEvent,
)


def test_new_public_objects_import_from_package() -> None:
    assert AIResponse is ai_api_guard.AIResponse
    assert AIProvider is ai_api_guard.AIProvider
    assert AIGuardError is ai_api_guard.AIGuardError
    assert ProviderError is ai_api_guard.ProviderError
    assert AuthenticationError is ai_api_guard.AuthenticationError
    assert RateLimitError is ai_api_guard.RateLimitError
    assert ProviderTimeoutError is ai_api_guard.ProviderTimeoutError
    assert ProviderUnavailableError is ai_api_guard.ProviderUnavailableError
    assert RetryEvent is ai_api_guard.RetryEvent
    assert "RetryEvent" in ai_api_guard.__all__
