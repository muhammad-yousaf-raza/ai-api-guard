"""Retry configuration for transient provider failures."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Immutable retry settings for transient provider failures.

    ``max_attempts`` is the total number of provider calls, including the
    first attempt. The delay after each failure grows exponentially from
    ``initial_delay`` and never exceeds ``max_delay``.
    """

    max_attempts: int = 3
    """Total provider calls allowed for one chat request."""

    initial_delay: float = 0.5
    """Delay, in seconds, after the first failed attempt."""

    max_delay: float = 8.0
    """Upper bound, in seconds, for any computed delay."""

    backoff_multiplier: float = 2.0
    """Factor applied to the delay after each additional failure."""

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
