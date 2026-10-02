"""Provider-independent models for AI API results."""

from dataclasses import dataclass
from typing import Any

from ai_api_guard.usage import TokenUsage, _require_non_negative_finite


@dataclass(frozen=True, slots=True)
class AIResponse:
    """Normalized result of a single AI API call.

    Holds generated text, the model that produced it, token counts, cost,
    and latency without depending on any provider SDK. ``raw_response`` keeps
    the original payload when a caller chooses to retain it.
    """

    content: str
    """Generated text returned by the call."""

    model: str
    """Identifier of the model that produced this response."""

    input_tokens: int = 0
    """Tokens consumed by the request."""

    output_tokens: int = 0
    """Tokens produced in the response."""

    cost: float = 0.0
    """Monetary cost of the call."""

    latency: float = 0.0
    """Elapsed time of the call, in seconds."""

    raw_response: Any | None = None
    """Original provider payload, when one was captured."""

    def __post_init__(self) -> None:
        """Validate token counts, cost, and latency."""
        TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)
        _require_non_negative_finite(self.cost, "cost")
        _require_non_negative_finite(self.latency, "latency")

    @property
    def usage(self) -> TokenUsage:
        """Return the token counts as a :class:`~ai_api_guard.usage.TokenUsage`."""
        return TokenUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens)

    @property
    def total_tokens(self) -> int:
        """Return the sum of input and output tokens."""
        return self.usage.total_tokens
