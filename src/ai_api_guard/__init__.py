"""AI API Guard.

Reliability, observability, and cost-control toolkit for AI APIs.
"""

from ai_api_guard.exceptions import (
    AIGuardError,
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.guard import AIGuard
from ai_api_guard.metrics import RequestMetrics
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider
from ai_api_guard.retry import RetryEvent, RetryPolicy
from ai_api_guard.usage import ModelPricing, TokenUsage, calculate_cost

__version__ = "0.1.0"

__all__ = [
    "AIGuard",
    "AIGuardError",
    "AIProvider",
    "AIResponse",
    "AuthenticationError",
    "ModelPricing",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RateLimitError",
    "RequestMetrics",
    "RetryEvent",
    "RetryPolicy",
    "TokenUsage",
    "__version__",
    "calculate_cost",
]
