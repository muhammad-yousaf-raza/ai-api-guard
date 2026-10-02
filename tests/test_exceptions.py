"""Tests for the AI API Guard exception hierarchy."""

import pytest

from ai_api_guard.exceptions import (
    AIGuardError,
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

_PROVIDER_ERRORS = (
    AuthenticationError,
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.mark.parametrize("error_type", _PROVIDER_ERRORS)
def test_provider_specific_errors_inherit_provider_error(
    error_type: type[ProviderError],
) -> None:
    assert issubclass(error_type, ProviderError)


def test_provider_error_inherits_ai_guard_error() -> None:
    assert issubclass(ProviderError, AIGuardError)
