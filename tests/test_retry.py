"""Tests for retry schedule configuration."""

import pytest

from ai_api_guard import RetryPolicy


def test_default_retry_policy_values() -> None:
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.initial_delay == 0.5
    assert policy.max_delay == 8.0
    assert policy.backoff_multiplier == 2.0


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (1, 0.5),
        (2, 1.0),
        (3, 2.0),
        (4, 4.0),
        (5, 8.0),
        (6, 8.0),
    ],
)
def test_exponential_delay(attempt: int, expected: float) -> None:
    policy = RetryPolicy(
        initial_delay=0.5,
        max_delay=8.0,
        backoff_multiplier=2.0,
    )

    assert policy.delay_for_attempt(attempt) == expected


def test_delay_is_capped_by_max_delay() -> None:
    policy = RetryPolicy(initial_delay=1.0, max_delay=3.0, backoff_multiplier=2.0)

    assert policy.delay_for_attempt(3) == 3.0
    assert policy.delay_for_attempt(4) == 3.0


def test_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_negative_initial_delay() -> None:
    with pytest.raises(ValueError, match="initial_delay"):
        RetryPolicy(initial_delay=-0.1)


def test_negative_max_delay() -> None:
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(max_delay=-1.0)


def test_backoff_multiplier_below_one() -> None:
    with pytest.raises(ValueError, match="backoff_multiplier"):
        RetryPolicy(backoff_multiplier=0.5)


def test_initial_delay_greater_than_max_delay() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        RetryPolicy(initial_delay=2.0, max_delay=1.0)


def test_delay_for_attempt_rejects_attempt_below_one() -> None:
    policy = RetryPolicy()

    with pytest.raises(ValueError, match="attempt"):
        policy.delay_for_attempt(0)
