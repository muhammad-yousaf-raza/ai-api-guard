"""Provider-independent record of one guarded chat request."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Outcome of a single ``AIGuard.chat`` invocation.

    One record covers the whole call, including any retries. It does not
    include token or cost data.
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
