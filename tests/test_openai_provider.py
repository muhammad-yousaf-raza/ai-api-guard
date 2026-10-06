"""Tests for the optional OpenAI adapter. No network calls are made."""

import email.utils
import subprocess
import sys
import time
from typing import Any

import pytest

from ai_api_guard.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_api_guard.providers.openai import OpenAIProvider
from ai_api_guard.usage import ModelPricing, TokenUsage, calculate_cost

_MESSAGES = [{"role": "user", "content": "Hello"}]


class _Completions:
    """Fake ``chat.completions`` namespace."""

    def __init__(self, result: Any = None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def create(self, *, model: str, messages: list[dict[str, str]]) -> Any:
        self.calls.append((model, messages))
        if self.error is not None:
            raise self.error
        return self.result


class _Client:
    """Fake OpenAI client with only the chat completions call we use."""

    def __init__(self, completions: _Completions) -> None:
        self.chat = type("_Chat", (), {"completions": completions})()


class _Message:
    def __init__(self, content: Any) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: Any) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(
        self,
        content: Any,
        *,
        model: str | None = "response-model",
        usage: Any = None,
        include_model: bool = True,
    ) -> None:
        self.choices = [_Choice(content)]
        self.usage = usage
        if include_model:
            self.model = model


class _Usage:
    def __init__(self, **fields: int) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


def _provider(
    completion: _Completion,
    *,
    pricing: dict[str, ModelPricing] | None = None,
) -> tuple[OpenAIProvider, _Completions]:
    completions = _Completions(result=completion)
    provider = OpenAIProvider(client=_Client(completions), pricing=pricing)
    return provider, completions


def _sdk_error(error_type: type[BaseException]) -> BaseException:
    """Build an SDK exception without the real HTTP constructor."""
    error = error_type.__new__(error_type)
    Exception.__init__(error, "sdk failure")
    return error


def test_provider_name() -> None:
    provider, _completions = _provider(_Completion("hello"))

    assert provider.name == "openai"


def test_supplied_client_is_used() -> None:
    completion = _Completion("hello")
    provider, completions = _provider(completion)

    provider.chat(model="requested-model", messages=_MESSAGES)

    assert completions.calls == [("requested-model", _MESSAGES)]


def test_chat_forwards_model_and_messages() -> None:
    provider, completions = _provider(_Completion("hello"))
    messages = [{"role": "user", "content": "Hello"}]

    provider.chat(model="gpt-test", messages=messages)

    assert completions.calls == [("gpt-test", messages)]
    assert completions.calls[0][1] is messages


def test_first_choice_content_is_mapped() -> None:
    completion = _Completion("hello from openai")
    provider, _completions = _provider(completion)

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.content == "hello from openai"
    assert response.latency == 0.0


def test_none_content_becomes_empty_string() -> None:
    provider, _completions = _provider(_Completion(None))

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.content == ""


def test_response_model_is_used_when_present() -> None:
    provider, _completions = _provider(_Completion("hello", model="actual-model"))

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.model == "actual-model"


@pytest.mark.parametrize(
    ("model", "include_model"),
    [("", True), (None, False)],
)
def test_requested_model_is_fallback(model: str | None, include_model: bool) -> None:
    completion = _Completion("hello", model=model, include_model=include_model)
    provider, _completions = _provider(completion)

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.model == "requested-model"


def test_prompt_and_completion_tokens_are_mapped() -> None:
    completion = _Completion(
        "hello",
        usage=_Usage(prompt_tokens=11, completion_tokens=7),
    )
    provider, _completions = _provider(completion)

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.total_tokens == 18


def test_missing_usage_produces_zero_tokens() -> None:
    provider, _completions = _provider(_Completion("hello", usage=None))

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.input_tokens == 0
    assert response.output_tokens == 0


def test_partial_usage_fields_become_zero() -> None:
    provider, _completions = _provider(_Completion("hello", usage=_Usage(prompt_tokens=4)))

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.input_tokens == 4
    assert response.output_tokens == 0


def test_raw_response_is_the_sdk_object() -> None:
    completion = _Completion("hello")
    provider, _completions = _provider(completion)

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.raw_response is completion


def test_actual_response_model_pricing_is_preferred() -> None:
    pricing = {
        "actual-model": ModelPricing(input_cost_per_million=1.0, output_cost_per_million=0.0),
        "requested-model": ModelPricing(input_cost_per_million=9.0, output_cost_per_million=0.0),
    }
    completion = _Completion(
        "hello",
        model="actual-model",
        usage=_Usage(prompt_tokens=1_000_000, completion_tokens=0),
    )
    provider, _completions = _provider(completion, pricing=pricing)

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.cost == calculate_cost(
        TokenUsage(input_tokens=1_000_000, output_tokens=0),
        pricing["actual-model"],
    )


def test_requested_model_pricing_is_fallback() -> None:
    pricing = {
        "requested-model": ModelPricing(input_cost_per_million=9.0, output_cost_per_million=0.0),
    }
    completion = _Completion(
        "hello",
        model="actual-model",
        usage=_Usage(prompt_tokens=1_000_000, completion_tokens=0),
    )
    provider, _completions = _provider(completion, pricing=pricing)

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.cost == calculate_cost(
        TokenUsage(input_tokens=1_000_000, output_tokens=0),
        pricing["requested-model"],
    )


def test_missing_pricing_produces_zero_cost() -> None:
    completion = _Completion(
        "hello",
        usage=_Usage(prompt_tokens=1_000_000, completion_tokens=50),
    )
    provider, _completions = _provider(completion, pricing={"other-model": ModelPricing(3.0, 4.0)})

    response = provider.chat(model="requested-model", messages=_MESSAGES)

    assert response.cost == 0.0


def test_cost_matches_calculate_cost() -> None:
    price = ModelPricing(input_cost_per_million=2.5, output_cost_per_million=4.0)
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    completion = _Completion(
        "hello",
        model="priced-model",
        usage=_Usage(prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens),
    )
    provider, _completions = _provider(completion, pricing={"priced-model": price})

    response = provider.chat(model="priced-model", messages=_MESSAGES)

    assert response.cost == calculate_cost(usage, price)


def test_provider_does_not_hard_code_prices() -> None:
    completion = _Completion(
        "hello",
        model="gpt-4o",
        usage=_Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000),
    )
    unpriced, _completions = _provider(completion)
    low_price, _low_calls = _provider(
        completion,
        pricing={"gpt-4o": ModelPricing(1.0, 1.0)},
    )
    high_price, _high_calls = _provider(
        completion,
        pricing={"gpt-4o": ModelPricing(8.0, 8.0)},
    )

    assert unpriced.chat(model="gpt-4o", messages=_MESSAGES).cost == 0.0
    assert low_price.chat(model="gpt-4o", messages=_MESSAGES).cost == 2.0
    assert high_price.chat(model="gpt-4o", messages=_MESSAGES).cost == 16.0


def test_sdk_errors_are_translated() -> None:
    openai = pytest.importorskip("openai")
    cases = (
        (openai.AuthenticationError, AuthenticationError, "authentication"),
        (openai.RateLimitError, RateLimitError, "rate limit"),
        (openai.APITimeoutError, ProviderTimeoutError, "timed out"),
        (openai.APIConnectionError, ProviderUnavailableError, "connection"),
        (openai.APIError, ProviderError, "API request failed"),
    )

    for sdk_type, guard_type, message in cases:
        original = _sdk_error(sdk_type)
        completions = _Completions(error=original)
        provider = OpenAIProvider(client=_Client(completions))

        with pytest.raises(guard_type, match=message) as caught:
            provider.chat(model="requested-model", messages=_MESSAGES)

        assert type(caught.value) is guard_type
        assert caught.value.__cause__ is original
        assert caught.value.retry_after is None


class _Headers:
    """Minimal header map with the ``get`` method the adapter reads."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _Response:
    """Minimal SDK response that exposes headers only."""

    def __init__(self, headers: _Headers) -> None:
        self.headers = headers


def _rate_limit_error(headers: dict[str, str] | None) -> BaseException:
    openai = pytest.importorskip("openai")
    error = _sdk_error(openai.RateLimitError)
    if headers is not None:
        error.response = _Response(_Headers(headers))
    return error


def _chat_rate_limit(headers: dict[str, str] | None) -> RateLimitError:
    original = _rate_limit_error(headers)
    provider = OpenAIProvider(client=_Client(_Completions(error=original)))

    with pytest.raises(RateLimitError, match="rate limit") as caught:
        provider.chat(model="requested-model", messages=_MESSAGES)

    assert type(caught.value) is RateLimitError
    assert caught.value.__cause__ is original
    assert "response" not in vars(caught.value)
    return caught.value


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "2"}, 2.0),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after-ms": "soon", "retry-after": "4"}, 4.0),
        ({}, None),
        ({"retry-after": "soon"}, None),
        ({"retry-after": "-5"}, None),
        ({"retry-after": "nan"}, None),
        ({"retry-after": "inf"}, None),
        ({"retry-after": "-inf"}, None),
        ({"retry-after-ms": "-1", "retry-after": "5"}, None),
    ],
)
def test_openai_rate_limit_retry_after(
    headers: dict[str, str],
    expected: float | None,
) -> None:
    error = _chat_rate_limit(headers)

    assert error.retry_after == expected


def test_openai_rate_limit_without_response_has_no_retry_after() -> None:
    error = _chat_rate_limit(None)

    assert error.retry_after is None


def test_openai_future_http_date_is_a_positive_delay() -> None:
    future = email.utils.formatdate(time.time() + 30, usegmt=True)
    error = _chat_rate_limit({"retry-after": future})

    assert error.retry_after is not None
    assert 0.0 < error.retry_after <= 30.0


def test_openai_past_http_date_waits_zero_seconds() -> None:
    past = email.utils.formatdate(time.time() - 30, usegmt=True)
    error = _chat_rate_limit({"retry-after": past})

    assert error.retry_after == 0.0


def test_http_date_retry_after_uses_supplied_clock() -> None:
    from ai_api_guard.providers.openai import _retry_after_from_headers

    headers = {"retry-after": "Thu, 01 Jan 1970 00:00:30 GMT"}

    assert _retry_after_from_headers(headers, now=0.0) == 30.0
    assert _retry_after_from_headers(headers, now=30.0) == 0.0
    assert _retry_after_from_headers(headers, now=31.0) == 0.0


def test_non_openai_errors_are_not_translated() -> None:
    original = ValueError("caller bug")
    completions = _Completions(error=original)
    provider = OpenAIProvider(client=_Client(completions))

    with pytest.raises(ValueError, match="caller bug") as caught:
        provider.chat(model="requested-model", messages=_MESSAGES)

    assert caught.value is original


def test_core_package_imports_without_openai_sdk() -> None:
    script = """
import sys

class _BlockOpenAI:
    def find_spec(self, fullname, path, target=None):
        if fullname == "openai" or fullname.startswith("openai."):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, _BlockOpenAI())
import ai_api_guard
from ai_api_guard import AIGuard, AIResponse, RetryPolicy
from ai_api_guard.providers import AIProvider
assert not hasattr(ai_api_guard, "OpenAIProvider")
assert not hasattr(sys.modules["ai_api_guard.providers"], "OpenAIProvider")
assert AIProvider.__name__ == "AIProvider"
assert AIGuard.__name__ == "AIGuard"
assert AIResponse.__name__ == "AIResponse"
assert RetryPolicy.__name__ == "RetryPolicy"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_sdk_explains_optional_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai" or name.startswith("openai."):
            raise ModuleNotFoundError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(ImportError, match=r'pip install "ai-api-guard\[openai\]"'):
        OpenAIProvider()
