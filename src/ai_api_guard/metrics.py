"""Provider-independent record of one guarded chat request."""

from dataclasses import dataclass

from ai_api_guard.usage import TokenUsage, _require_non_negative_finite


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Outcome of a single ``AIGuard.chat`` invocation.

    One record covers the whole call, including any retries. On success, token
    counts and cost are copied from the response. A failed call records zeros.
    """

    provider: str
    """Name of the provider that handled the request."""

    model: str
    """Model identifier supplied by the caller."""

    attempts: int
    """Total provider calls made for this request."""

    latency: float
    """Elapsed time from the first attempt through the final outcome, in seconds."""

    success: bool
    """Whether the request returned a response."""

    error_type: str | None = None
    """Exception class name when the request failed."""

    input_tokens: int = 0
    """Input tokens from the successful response, or zero after a failure."""

    output_tokens: int = 0
    """Output tokens from the successful response, or zero after a failure."""

    cost: float = 0.0
    """Cost copied from the successful response, or zero after a failure."""

    def __post_init__(self) -> None:
        """Reject records that cannot describe a finished request."""
        if self.provider == "":
            message = "provider must not be empty"
            raise ValueError(message)
        if self.model == "":
            message = "model must not be empty"
            raise ValueError(message)
        if self.attempts < 1:
            message = f"attempts must be >= 1, got {self.attempts}"
            raise ValueError(message)
        if self.latency < 0:
            message = f"latency must be >= 0, got {self.latency}"
            raise ValueError(message)
        if self.success and self.error_type is not None:
            message = "success=True requires error_type to be None"
            raise ValueError(message)
        if not self.success and self.error_type is None:
            message = "success=False requires error_type"
            raise ValueError(message)
        if not self.success and self.error_type == "":
            message = "success=False requires a non-empty error_type"
            raise ValueError(message)
        TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)
        _require_non_negative_finite(self.cost, "cost")

    @property
    def total_tokens(self) -> int:
        """Return the sum of input and output tokens."""
        return self.input_tokens + self.output_tokens
