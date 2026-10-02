"""Provider-independent token usage and caller-supplied pricing."""

import math
from dataclasses import dataclass

_TOKENS_PER_MILLION = 1_000_000


def _require_non_negative_finite(value: float, name: str) -> None:
    """Reject NaN, infinities, and negative currency or duration values."""
    if math.isnan(value):
        message = f"{name} must not be NaN"
        raise ValueError(message)
    if math.isinf(value):
        sign = "positive" if value > 0 else "negative"
        message = f"{name} must not be {sign} infinity"
        raise ValueError(message)
    if value < 0:
        message = f"{name} must be >= 0, got {value}"
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one model call.

    This is the canonical usage value. It does not include a provider, model
    name, or price.
    """

    input_tokens: int = 0
    """Tokens consumed by the request."""

    output_tokens: int = 0
    """Tokens produced in the response."""

    def __post_init__(self) -> None:
        """Reject negative token counts."""
        if self.input_tokens < 0:
            message = f"input_tokens must be >= 0, got {self.input_tokens}"
            raise ValueError(message)
        if self.output_tokens < 0:
            message = f"output_tokens must be >= 0, got {self.output_tokens}"
            raise ValueError(message)

    @property
    def total_tokens(self) -> int:
        """Return the sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Caller-supplied price for one million input or output tokens.

    Values are currency units per 1,000,000 tokens. The currency is not
    specified here, and this object does not name a provider or model.
    """

    input_cost_per_million: float
    """Currency units charged per 1,000,000 input tokens."""

    output_cost_per_million: float
    """Currency units charged per 1,000,000 output tokens."""

    def __post_init__(self) -> None:
        """Reject negative or non-finite prices."""
        _require_non_negative_finite(
            self.input_cost_per_million,
            "input_cost_per_million",
        )
        _require_non_negative_finite(
            self.output_cost_per_million,
            "output_cost_per_million",
        )


def calculate_cost(usage: TokenUsage, pricing: ModelPricing) -> float:
    """Return the unrounded cost of ``usage`` at ``pricing``.

    Input and output tokens are priced separately per 1,000,000 tokens and
    then added. The arguments are not modified.

    Args:
        usage: Token counts to price.
        pricing: Caller-supplied prices per million tokens.

    Returns:
        The combined cost in the same currency units as ``pricing``.
    """
    input_cost = usage.input_tokens / _TOKENS_PER_MILLION * pricing.input_cost_per_million
    output_cost = usage.output_tokens / _TOKENS_PER_MILLION * pricing.output_cost_per_million
    return input_cost + output_cost
