"""Public entry point that delegates chat requests to an AI provider."""

from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider


class AIGuard:
    """Main entry point for guarded AI provider requests."""

    def __init__(self, provider: AIProvider) -> None:
        """Store the provider that handles chat requests.

        Args:
            provider: Provider implementation used for every chat call.
        """
        self._provider = provider

    @property
    def provider(self) -> AIProvider:
        """Return the provider this guard delegates to."""
        return self._provider

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        """Forward a chat request to the configured provider.

        The model, messages, and returned response are left unchanged.
        Provider exceptions propagate to the caller.

        Args:
            model: Model identifier understood by the provider.
            messages: Ordered chat messages. Each mapping uses string keys and
                string values.

        Returns:
            The response object returned by the provider.
        """
        return self._provider.chat(model=model, messages=messages)
