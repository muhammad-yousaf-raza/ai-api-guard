"""Tests for the AI API Guard exception hierarchy."""

import math

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


_RETRYABLE_ERRORS = (
    RateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.mark.parametrize("error_type", _PROVIDER_ERRORS)
def test_provider_errors_default_retry_after_to_none(
    error_type: type[ProviderError],
) -> None:
    error = error_type("limited")

    assert error.retry_after is None
    assert error.args == ("limited",)
    assert str(error) == "limited"


@pytest.mark.parametrize("error_type", _RETRYABLE_ERRORS)
def test_retryable_errors_normalize_integer_retry_after(
    error_type: type[ProviderError],
) -> None:
    error = error_type("limited", retry_after=2)

    assert error.retry_after == 2.0
    assert isinstance(error.retry_after, float)
    assert str(error) == "limited"


def test_retry_after_zero_is_valid() -> None:
    assert RateLimitError("limited", retry_after=0).retry_after == 0.0
    assert RateLimitError("limited", retry_after=0.0).retry_after == 0.0


def test_explicit_none_retry_after_stays_none() -> None:
    error = ProviderTimeoutError("slow", retry_after=None)

    assert error.retry_after is None


@pytest.mark.parametrize("value", [-0.1, -1, math.nan, math.inf, -math.inf])
def test_retry_after_rejects_negative_or_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="retry_after"):
        RateLimitError("limited", retry_after=value)


@pytest.mark.parametrize("value", [True, False, "1", b"1", object()])
def test_retry_after_rejects_unsupported_types(value: object) -> None:
    with pytest.raises(TypeError, match="retry_after"):
        ProviderUnavailableError("down", retry_after=value)  # type: ignore[arg-type]
