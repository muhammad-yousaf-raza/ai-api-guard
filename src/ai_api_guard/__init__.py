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
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider
from ai_api_guard.retry import RetryPolicy

__version__ = "0.1.0"

__all__ = [
    "AIGuard",
    "AIGuardError",
    "AIProvider",
    "AIResponse",
    "AuthenticationError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RateLimitError",
    "RetryPolicy",
    "__version__",
]
