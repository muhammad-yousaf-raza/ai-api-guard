"""Optional OpenAI chat adapter.

Importing this module does not import the OpenAI SDK. The SDK is imported
only when a client must be constructed or an SDK exception must be translated.
"""

import email.utils
import math
import time
from collections.abc import Mapping
from typing import Any, Literal

from ai_api_guard.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.models import AIResponse
from ai_api_guard.providers.base import AIProvider
from ai_api_guard.usage import ModelPricing, TokenUsage, calculate_cost

_INSTALL_HINT = 'pip install "ai-api-guard[openai]"'
_MILLISECONDS_PER_SECOND = 1000.0


class OpenAIProvider(AIProvider):
    """Chat adapter for the official OpenAI Python SDK.

    Prices are never built in. Pass a ``pricing`` map of model name to
    :class:`~ai_api_guard.usage.ModelPricing` when a call should report cost.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        pricing: Mapping[str, ModelPricing] | None = None,
    ) -> None:
        """Store an OpenAI client and optional caller-supplied prices.

        Args:
            api_key: API key used when this provider constructs the SDK client.
                Ignored when ``client`` is supplied.
            client: Existing OpenAI SDK client. When omitted, one is created
                with the optional OpenAI dependency.
            pricing: Model name to price per million tokens. Keys may be the
                requested model or the model name returned by the API.
        """
        self._client = _build_openai_client(api_key) if client is None else client
        self._pricing = {} if pricing is None else dict(pricing)

    @property
    def name(self) -> str:
        """Return the stable provider identifier."""
        return "openai"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        """Send a chat completion request and normalize the SDK response.

        Args:
            model: Model identifier requested from OpenAI.
            messages: Ordered chat messages with string keys and values.

        Returns:
            The normalized response. Latency is left at the model default so
            :class:`~ai_api_guard.guard.AIGuard` can record request latency.
        """
        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as exc:
            translated = _translate_openai_error(exc)
            if translated is None:
                raise
            raise translated from exc
        return _to_ai_response(
            completion,
            requested_model=model,
            pricing=self._pricing,
        )


def _build_openai_client(api_key: str | None) -> Any:
    """Create an official OpenAI client, or explain how to install the SDK."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        message = f"OpenAI support is not installed. Install it with: {_INSTALL_HINT}"
        raise ImportError(message) from exc
    return OpenAI(api_key=api_key)


def _translate_openai_error(exc: BaseException) -> ProviderError | None:
    """Map a recognized OpenAI SDK error to a provider-independent error.

    Unrelated exceptions return ``None`` so the caller can re-raise them
    unchanged. More specific SDK errors are matched before ``APIError``.
    """
    try:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
        )
        from openai import (
            AuthenticationError as OpenAIAuthenticationError,
        )
        from openai import (
            RateLimitError as OpenAIRateLimitError,
        )
    except ImportError:
        return None

    if isinstance(exc, OpenAIAuthenticationError):
        return AuthenticationError("OpenAI authentication failed")
    if isinstance(exc, OpenAIRateLimitError):
        return RateLimitError(
            "OpenAI rate limit exceeded",
            retry_after=_retry_after_from_openai_error(exc),
        )
    if isinstance(exc, APITimeoutError):
        return ProviderTimeoutError("OpenAI request timed out")
    if isinstance(exc, APIConnectionError):
        return ProviderUnavailableError("OpenAI connection failed")
    if isinstance(exc, APIError):
        return ProviderError("OpenAI API request failed")
    return None


def _retry_after_from_openai_error(exc: BaseException) -> float | None:
    """Read a Retry-After delay from an OpenAI error, in seconds.

    Missing or unusable header values return ``None``. This never attaches
    the SDK response to the translated error.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return _retry_after_from_headers(headers, now=time.time())


def _retry_after_from_headers(headers: Any, *, now: float) -> float | None:
    """Parse Retry-After headers into seconds.

    ``retry-after-ms`` wins when it is a usable number. A non-numeric
    ``retry-after-ms`` falls through to ``retry-after``. A negative,
    ``NaN``, or infinite numeric value returns ``None`` instead of falling
    through. ``now`` is seconds since the Unix epoch and is used only for
    HTTP-date values.
    """
    milliseconds = _header_value(headers, "retry-after-ms")
    if milliseconds is not None:
        parsed = _non_negative_seconds(milliseconds, divisor=_MILLISECONDS_PER_SECOND)
        if parsed != "malformed":
            return None if parsed == "invalid" else parsed

    raw_seconds = _header_value(headers, "retry-after")
    if raw_seconds is None:
        return None
    parsed = _non_negative_seconds(raw_seconds, divisor=1.0)
    if parsed == "malformed" and isinstance(raw_seconds, str):
        return _http_date_delay(raw_seconds, now=now)
    if parsed == "malformed" or parsed == "invalid":
        return None
    return parsed


def _header_value(headers: Any, name: str) -> object | None:
    """Return one header value, or None when it is absent."""
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    try:
        value: object = getter(name)
    except (AttributeError, TypeError):
        return None
    return value


def _non_negative_seconds(
    raw: object,
    *,
    divisor: float,
) -> float | Literal["malformed", "invalid"]:
    """Convert a header value to seconds, or a short failure tag."""
    if isinstance(raw, str):
        try:
            number = float(raw.strip())
        except ValueError:
            return "malformed"
    elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return "malformed"
    else:
        number = float(raw)
    if math.isnan(number) or math.isinf(number) or number < 0:
        return "invalid"
    return number / divisor


def _http_date_delay(value: str, *, now: float) -> float | None:
    """Convert an HTTP-date Retry-After header into a non-negative delay."""
    try:
        parsed = email.utils.parsedate_tz(value)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    if parsed is None:
        return None
    try:
        moment = float(email.utils.mktime_tz(parsed))
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    delay = moment - now
    if math.isnan(delay) or math.isinf(delay):
        return None
    if delay < 0:
        return 0.0
    return delay


def _to_ai_response(
    completion: Any,
    *,
    requested_model: str,
    pricing: Mapping[str, ModelPricing],
) -> AIResponse:
    """Copy the first choice, token usage, and optional cost into AIResponse."""
    raw_content = completion.choices[0].message.content
    content = raw_content if isinstance(raw_content, str) else ""
    model = _resolved_model(completion, requested_model)
    usage = _token_usage(getattr(completion, "usage", None))
    return AIResponse(
        content=content,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost=_cost_for(model, requested_model, usage, pricing),
        raw_response=completion,
    )


def _resolved_model(completion: Any, requested_model: str) -> str:
    """Prefer a non-empty model name from the SDK response."""
    response_model = getattr(completion, "model", None)
    if isinstance(response_model, str) and response_model != "":
        return response_model
    return requested_model


def _token_usage(usage: Any) -> TokenUsage:
    """Read prompt and completion tokens, treating gaps as zero."""
    return TokenUsage(
        input_tokens=_token_count(usage, "prompt_tokens"),
        output_tokens=_token_count(usage, "completion_tokens"),
    )


def _token_count(usage: Any, field_name: str) -> int:
    """Return one token field, or zero when it is absent or not an integer."""
    if usage is None:
        return 0
    value = getattr(usage, field_name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _cost_for(
    response_model: str,
    requested_model: str,
    usage: TokenUsage,
    pricing: Mapping[str, ModelPricing],
) -> float:
    """Price tokens from the response model, then the requested model."""
    model_pricing = pricing.get(response_model)
    if model_pricing is None:
        model_pricing = pricing.get(requested_model)
    if model_pricing is None:
        return 0.0
    return calculate_cost(usage, model_pricing)
