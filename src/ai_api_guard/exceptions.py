"""Errors raised by AI API Guard."""


class AIGuardError(Exception):
    """Base error for failures raised by AI API Guard."""


class ProviderError(AIGuardError):
    """Base error for failures that originate at an AI provider."""


class AuthenticationError(ProviderError):
    """The provider rejected the caller's credentials."""


class RateLimitError(ProviderError):
    """The provider refused the call because a rate limit was exceeded."""


class ProviderTimeoutError(ProviderError):
    """The provider did not answer before the call timed out."""


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached or is temporarily unavailable."""
