"""Tests for retry schedule configuration."""

import math
from collections.abc import Callable

import pytest

from ai_api_guard import RetryPolicy


def test_default_retry_policy_values() -> None:
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.initial_delay == 0.5
    assert policy.max_delay == 8.0
    assert policy.backoff_multiplier == 2.0
    assert policy.jitter == 0.0


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


def test_missing_retry_after_uses_exponential_delay() -> None:
    policy = RetryPolicy(initial_delay=0.5, max_delay=8.0, backoff_multiplier=2.0)

    assert policy.delay_for_retry(1, retry_after=None) == policy.delay_for_attempt(1)
    assert policy.delay_for_retry(2, retry_after=None) == 1.0


def test_retry_after_replaces_exponential_delay() -> None:
    policy = RetryPolicy(initial_delay=0.5, max_delay=8.0, backoff_multiplier=2.0)

    assert policy.delay_for_retry(1, retry_after=0.25) == 0.25
    assert policy.delay_for_retry(2, retry_after=3) == 3.0


def test_retry_after_is_capped_by_max_delay() -> None:
    policy = RetryPolicy(initial_delay=0.5, max_delay=8.0)

    assert policy.delay_for_retry(1, retry_after=30) == 8.0
    assert policy.delay_for_retry(1, retry_after=8) == 8.0


def test_zero_retry_after_waits_zero_seconds() -> None:
    policy = RetryPolicy(initial_delay=0.5, max_delay=8.0)

    assert policy.delay_for_retry(1, retry_after=0.0) == 0.0


def test_retry_after_rejects_invalid_attempt() -> None:
    policy = RetryPolicy()

    with pytest.raises(ValueError, match="attempt"):
        policy.delay_for_retry(0, retry_after=1.0)


def _forbidden_random() -> float:
    message = "random source must not be called"
    raise AssertionError(message)


def _constant_random(value: object) -> Callable[[], float]:
    def draw() -> float:
        return value  # type: ignore[return-value]

    return draw


class _ScriptedRandom:
    """Return scripted unit draws in order."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self._values.pop(0)


def test_jitter_zero_does_not_call_random_source() -> None:
    policy = RetryPolicy(jitter=0, unit_random=_forbidden_random)

    assert policy.jitter == 0.0
    assert isinstance(policy.jitter, float)
    assert policy.delay_for_retry(1) == 0.5
    assert policy.delay_for_retry(2) == 1.0


def test_delay_for_attempt_ignores_jitter() -> None:
    policy = RetryPolicy(jitter=1.0, unit_random=_forbidden_random)

    assert policy.delay_for_attempt(1) == 0.5
    assert policy.delay_for_attempt(2) == 1.0
    assert policy.delay_for_attempt(5) == 8.0


@pytest.mark.parametrize(
    ("draw", "expected"),
    [
        (0.0, 0.25),
        (0.5, 0.375),
        (1.0, 0.5),
    ],
)
def test_full_equal_jitter(draw: float, expected: float) -> None:
    policy = RetryPolicy(jitter=1.0, unit_random=_constant_random(draw))

    assert policy.delay_for_retry(1) == expected


@pytest.mark.parametrize(
    ("draw", "expected"),
    [
        (0.0, 0.375),
        (1.0, 0.5),
    ],
)
def test_partial_jitter(draw: float, expected: float) -> None:
    policy = RetryPolicy(jitter=0.5, unit_random=_constant_random(draw))

    assert policy.delay_for_retry(1) == expected


def test_jitter_uses_capped_exponential_base() -> None:
    policy = RetryPolicy(jitter=1, unit_random=_constant_random(0.0))
    upper = RetryPolicy(jitter=1, unit_random=_constant_random(1.0))

    assert policy.delay_for_attempt(5) == 8.0
    assert policy.delay_for_retry(5) == 4.0
    assert upper.delay_for_retry(5) == 8.0


def test_jitter_of_zero_base_stays_zero() -> None:
    policy = RetryPolicy(
        initial_delay=0.0,
        max_delay=8.0,
        jitter=1.0,
        unit_random=_constant_random(1.0),
    )

    assert policy.delay_for_retry(1) == 0.0


def test_scripted_random_draws_apply_per_attempt() -> None:
    random_source = _ScriptedRandom([0.0, 1.0])
    policy = RetryPolicy(jitter=1.0, unit_random=random_source)

    assert policy.delay_for_retry(1) == 0.25
    assert policy.delay_for_retry(2) == 1.0
    assert random_source.calls == 2


def test_retry_after_bypasses_jitter() -> None:
    policy = RetryPolicy(jitter=1.0, unit_random=_forbidden_random)

    assert policy.delay_for_retry(1, retry_after=3) == 3.0
    assert policy.delay_for_retry(1, retry_after=30) == 8.0
    assert policy.delay_for_retry(1, retry_after=0.0) == 0.0


def test_later_attempt_without_retry_after_is_jittered() -> None:
    random_source = _ScriptedRandom([0.0])
    policy = RetryPolicy(jitter=1.0, unit_random=random_source)

    assert policy.delay_for_retry(1, retry_after=3.0) == 3.0
    assert random_source.calls == 0
    assert policy.delay_for_retry(2) == 0.5
    assert random_source.calls == 1


def test_jittered_retry_rejects_attempt_below_one() -> None:
    policy = RetryPolicy(jitter=1.0, unit_random=_forbidden_random)

    with pytest.raises(ValueError, match="attempt"):
        policy.delay_for_retry(0)
    with pytest.raises(ValueError, match="attempt"):
        policy.delay_for_retry(0, retry_after=1.0)


def test_integer_jitter_is_normalized() -> None:
    policy = RetryPolicy(jitter=1)

    assert policy.jitter == 1.0
    assert isinstance(policy.jitter, float)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_jitter_rejects_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match="jitter"):
        RetryPolicy(jitter=value)


@pytest.mark.parametrize("value", [True, False, "1", b"1", object()])
def test_jitter_rejects_unsupported_types(value: object) -> None:
    with pytest.raises(TypeError, match="jitter"):
        RetryPolicy(jitter=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_random_draw_rejects_out_of_range_values(value: float) -> None:
    policy = RetryPolicy(jitter=1.0, unit_random=_constant_random(value))

    with pytest.raises(ValueError, match="random draw"):
        policy.delay_for_retry(1)


@pytest.mark.parametrize("value", [True, False, "0.5", b"0", object()])
def test_random_draw_rejects_unsupported_types(value: object) -> None:
    policy = RetryPolicy(jitter=1.0, unit_random=_constant_random(value))

    with pytest.raises(TypeError, match="random draw"):
        policy.delay_for_retry(1)


def test_random_source_is_excluded_from_repr_and_equality() -> None:
    left = RetryPolicy(jitter=1.0, unit_random=_constant_random(0.0))
    right = RetryPolicy(jitter=1.0, unit_random=_constant_random(1.0))

    assert left == right
    assert "unit_random" not in repr(left)
