"""Retry configuration for transient provider failures."""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from ai_api_guard.exceptions import _validate_retry_after


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Immutable retry settings for transient provider failures.

    ``max_attempts`` is the total number of provider calls, including the
    first attempt. The delay after each failure grows exponentially from
    ``initial_delay`` and never exceeds ``max_delay``. A server-supplied
    delay can replace that calculation for one attempt. ``jitter`` is off
    by default. When it is enabled, equal jitter can shorten only the
    exponential delay.
    """

    max_attempts: int = 3
    """Total provider calls allowed for one chat request."""

    initial_delay: float = 0.5
    """Delay, in seconds, after the first failed attempt."""

    max_delay: float = 8.0
    """Upper bound, in seconds, for any computed delay."""

    backoff_multiplier: float = 2.0
    """Factor applied to the delay after each additional failure."""

    jitter: float = 0.0
    """Equal-jitter strength from 0 to 1. ``0`` leaves delays unchanged."""

    unit_random: Callable[[], float] = field(
        default=random.random,
        compare=False,
        hash=False,
        repr=False,
    )
    """Return one finite float in ``[0, 1]`` when jitter needs a draw."""

    def __post_init__(self) -> None:
        """Reject configuration that cannot describe a valid retry schedule."""
        if self.max_attempts < 1:
            message = f"max_attempts must be >= 1, got {self.max_attempts}"
            raise ValueError(message)
        if self.initial_delay < 0:
            message = f"initial_delay must be >= 0, got {self.initial_delay}"
            raise ValueError(message)
        if self.max_delay < 0:
            message = f"max_delay must be >= 0, got {self.max_delay}"
            raise ValueError(message)
        if self.backoff_multiplier < 1:
            message = f"backoff_multiplier must be >= 1, got {self.backoff_multiplier}"
            raise ValueError(message)
        if self.initial_delay > self.max_delay:
            message = (
                "initial_delay must not exceed max_delay, "
                f"got initial_delay={self.initial_delay} and max_delay={self.max_delay}"
            )
            raise ValueError(message)
        object.__setattr__(self, "jitter", _validate_jitter(self.jitter))

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the delay, in seconds, after a failed attempt.

        ``attempt`` is 1-based. The first failure waits ``initial_delay``.
        Each later failure multiplies the previous delay by
        ``backoff_multiplier``. The result is capped at ``max_delay``.

        Args:
            attempt: The failed attempt that just occurred. Must be >= 1.

        Returns:
            The delay to wait before the next attempt.
        """
        if attempt < 1:
            message = f"attempt must be >= 1, got {attempt}"
            raise ValueError(message)

        delay = self.initial_delay
        for _ in range(attempt - 1):
            if delay >= self.max_delay:
                return self.max_delay
            delay *= self.backoff_multiplier
        if delay > self.max_delay:
            return self.max_delay
        return delay

    def delay_for_retry(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Return the delay, in seconds, before the next attempt.

        A server-supplied ``retry_after`` replaces the exponential delay for
        this attempt. It is not added to that delay, it is not jittered, and
        it is not combined with :meth:`delay_for_attempt` by taking the
        greater value. The result never exceeds ``max_delay``. ``None`` uses
        the exponential schedule. When ``jitter`` is greater than zero, that
        exponential delay is spread downward by equal jitter. ``0.0`` waits
        for zero seconds.

        Args:
            attempt: The failed attempt that just occurred. Must be >= 1.
            retry_after: Server-requested delay in seconds, or ``None``.

        Returns:
            The delay to wait before the next attempt.
        """
        validated = _validate_retry_after(retry_after)
        if validated is not None:
            if attempt < 1:
                message = f"attempt must be >= 1, got {attempt}"
                raise ValueError(message)
            if validated > self.max_delay:
                return self.max_delay
            return validated
        base = self.delay_for_attempt(attempt)
        if self.jitter == 0.0:
            return base
        draw = _validate_unit_draw(self.unit_random())
        span = base * (self.jitter / 2.0)
        return (base - span) + (span * draw)


def _finite_number(value: object, name: str) -> float:
    """Return a finite float, rejecting booleans and other types."""
    if isinstance(value, bool):
        message = f"{name} must be a number from 0 to 1, got bool"
        raise TypeError(message)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value):
            message = f"{name} must not be NaN"
            raise ValueError(message)
        if math.isinf(value):
            sign = "positive" if value > 0 else "negative"
            message = f"{name} must not be {sign} infinity"
            raise ValueError(message)
        return value
    message = f"{name} must be a number from 0 to 1, got {type(value).__name__}"
    raise TypeError(message)


def _validate_jitter(value: object) -> float:
    """Return jitter strength in ``[0, 1]``."""
    number = _finite_number(value, "jitter")
    if number < 0 or number > 1:
        message = f"jitter must be from 0 to 1, got {number}"
        raise ValueError(message)
    return number


def _validate_unit_draw(value: object) -> float:
    """Return a random draw in ``[0, 1]``."""
    number = _finite_number(value, "random draw")
    if number < 0 or number > 1:
        message = f"random draw must be from 0 to 1, got {number}"
        raise ValueError(message)
    return number
