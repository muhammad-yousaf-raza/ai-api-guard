"""Provider-independent interface for AI chat APIs."""

from abc import ABC, abstractmethod

from ai_api_guard.models import AIResponse


class AIProvider(ABC):
    """Contract for a chat-capable AI provider.

    Concrete providers turn a provider-neutral message list into an
    :class:`~ai_api_guard.models.AIResponse`. This package does not include
    a built-in provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable identifier for this provider."""

    @abstractmethod
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        """Send a chat request and return a normalized response.

        Arguments are keyword-only so callers name ``model`` and ``messages``
        explicitly.

        Args:
            model: Model identifier understood by the provider.
            messages: Ordered chat messages. Each mapping uses string keys and
                string values.

        Returns:
            The normalized result of the call.
        """
