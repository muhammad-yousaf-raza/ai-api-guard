"""Tests for token usage, caller-supplied pricing, and cost calculation."""

import math

import pytest

from ai_api_guard import (
    AIGuard,
    ModelPricing,
    RequestMetrics,
    RetryPolicy,
    TokenUsage,
    calculate_cost,
)
from ai_api_guard.exceptions import AuthenticationError, RateLimitError
from ai_api_guard.models import AIResponse
from ai_api_guard.providers import AIProvider


def test_token_usage_defaults_to_zero() -> None:
    usage = TokenUsage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_token_usage_total_tokens() -> None:
    usage = TokenUsage(input_tokens=3, output_tokens=5)

    assert usage.total_tokens == 8


def test_negative_input_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        TokenUsage(input_tokens=-1)


def test_negative_output_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="output_tokens"):
        TokenUsage(output_tokens=-1)


def test_valid_pricing_accepted() -> None:
    pricing = ModelPricing(input_cost_per_million=0.15, output_cost_per_million=0.6)

    assert pricing.input_cost_per_million == 0.15
    assert pricing.output_cost_per_million == 0.6


def test_zero_pricing_accepted() -> None:
    pricing = ModelPricing(input_cost_per_million=0.0, output_cost_per_million=0.0)

    assert pricing.input_cost_per_million == 0.0
    assert pricing.output_cost_per_million == 0.0


def test_negative_input_price_rejected() -> None:
    with pytest.raises(ValueError, match="input_cost_per_million"):
        ModelPricing(input_cost_per_million=-1.0, output_cost_per_million=1.0)


def test_negative_output_price_rejected() -> None:
    with pytest.raises(ValueError, match="output_cost_per_million"):
        ModelPricing(input_cost_per_million=1.0, output_cost_per_million=-1.0)


@pytest.mark.parametrize(
    "field",
    ["input_cost_per_million", "output_cost_per_million"],
)
def test_nan_pricing_rejected(field: str) -> None:
    values = {"input_cost_per_million": 1.0, "output_cost_per_million": 2.0}
    values[field] = math.nan

    with pytest.raises(ValueError, match="NaN"):
        ModelPricing(**values)


@pytest.mark.parametrize(
    "field",
    ["input_cost_per_million", "output_cost_per_million"],
)
def test_positive_infinity_pricing_rejected(field: str) -> None:
    values = {"input_cost_per_million": 1.0, "output_cost_per_million": 2.0}
    values[field] = math.inf

    with pytest.raises(ValueError, match="positive infinity"):
        ModelPricing(**values)


@pytest.mark.parametrize(
    "field",
    ["input_cost_per_million", "output_cost_per_million"],
)
def test_negative_infinity_pricing_rejected(field: str) -> None:
    values = {"input_cost_per_million": 1.0, "output_cost_per_million": 2.0}
    values[field] = -math.inf

    with pytest.raises(ValueError, match="negative infinity"):
        ModelPricing(**values)


def test_zero_usage_cost_is_zero() -> None:
    result = calculate_cost(TokenUsage(), ModelPricing(3.0, 4.0))

    assert result == 0.0


def test_input_only_cost() -> None:
    result = calculate_cost(
        TokenUsage(input_tokens=1_000_000, output_tokens=0),
        ModelPricing(input_cost_per_million=3.0, output_cost_per_million=9.0),
    )

    assert math.isclose(result, 3.0)


def test_output_only_cost() -> None:
    result = calculate_cost(
        TokenUsage(input_tokens=0, output_tokens=500_000),
        ModelPricing(input_cost_per_million=9.0, output_cost_per_million=4.0),
    )

    assert math.isclose(result, 2.0)


def test_combined_cost() -> None:
    result = calculate_cost(
        TokenUsage(input_tokens=1_000_000, output_tokens=2_000_000),
        ModelPricing(input_cost_per_million=2.5, output_cost_per_million=5.0),
    )

    assert math.isclose(result, 12.5)


def test_cost_is_not_rounded() -> None:
    result = calculate_cost(
        TokenUsage(input_tokens=1, output_tokens=0),
        ModelPricing(input_cost_per_million=1.0, output_cost_per_million=1.0),
    )

    assert math.isclose(result, 1 / 1_000_000)
    assert result != round(result, 2)


def test_calculate_cost_does_not_mutate_inputs() -> None:
    usage = TokenUsage(input_tokens=4, output_tokens=6)
    pricing = ModelPricing(input_cost_per_million=1.25, output_cost_per_million=2.5)
    before = (
        usage.input_tokens,
        usage.output_tokens,
        pricing.input_cost_per_million,
        pricing.output_cost_per_million,
    )

    calculate_cost(usage, pricing)

    assert (
        usage.input_tokens,
        usage.output_tokens,
        pricing.input_cost_per_million,
        pricing.output_cost_per_million,
    ) == before


def test_ai_response_token_access_remains_valid() -> None:
    response = AIResponse(
        content="hello",
        model="test-model",
        input_tokens=3,
        output_tokens=5,
    )

    assert response.input_tokens == 3
    assert response.output_tokens == 5
    assert response.total_tokens == 8
    assert response.usage == TokenUsage(input_tokens=3, output_tokens=5)


def test_ai_response_rejects_negative_input_tokens() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        AIResponse(content="hello", model="test-model", input_tokens=-1)


def test_ai_response_rejects_negative_output_tokens() -> None:
    with pytest.raises(ValueError, match="output_tokens"):
        AIResponse(content="hello", model="test-model", output_tokens=-1)


def test_ai_response_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="cost"):
        AIResponse(content="hello", model="test-model", cost=-0.01)


def test_ai_response_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latency"):
        AIResponse(content="hello", model="test-model", latency=-0.01)


def test_ai_response_rejects_nan_cost() -> None:
    with pytest.raises(ValueError, match="NaN"):
        AIResponse(content="hello", model="test-model", cost=math.nan)


@pytest.mark.parametrize("cost", [math.inf, -math.inf])
def test_ai_response_rejects_infinite_cost(cost: float) -> None:
    with pytest.raises(ValueError, match="infinity"):
        AIResponse(content="hello", model="test-model", cost=cost)


def test_ai_response_rejects_nan_latency() -> None:
    with pytest.raises(ValueError, match="NaN"):
        AIResponse(content="hello", model="test-model", latency=math.nan)


@pytest.mark.parametrize("latency", [math.inf, -math.inf])
def test_ai_response_rejects_infinite_latency(latency: float) -> None:
    with pytest.raises(ValueError, match="infinity"):
        AIResponse(content="hello", model="test-model", latency=latency)


def test_successful_metrics_accept_usage_and_cost() -> None:
    metrics = RequestMetrics(
        provider="echo",
        model="test-model",
        attempts=1,
        latency=0.1,
        success=True,
        input_tokens=4,
        output_tokens=6,
        cost=0.25,
    )

    assert metrics.input_tokens == 4
    assert metrics.output_tokens == 6
    assert math.isclose(metrics.cost, 0.25)
    assert metrics.total_tokens == 10


def test_metrics_reject_negative_tokens() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=True,
            input_tokens=-1,
        )
    with pytest.raises(ValueError, match="output_tokens"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=True,
            output_tokens=-1,
        )


@pytest.mark.parametrize("cost", [-0.1, math.nan, math.inf, -math.inf])
def test_metrics_reject_invalid_cost(cost: float) -> None:
    with pytest.raises(ValueError, match="cost"):
        RequestMetrics(
            provider="echo",
            model="test-model",
            attempts=1,
            latency=0.0,
            success=True,
            cost=cost,
        )


def test_failed_metrics_default_usage_and_cost_to_zero() -> None:
    metrics = RequestMetrics(
        provider="echo",
        model="test-model",
        attempts=1,
        latency=0.1,
        success=False,
        error_type="RateLimitError",
    )

    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.cost == 0.0
    assert metrics.total_tokens == 0


class _Clock:
    """Returns scripted timestamps."""

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0)


class _Sink:
    """Stores metrics records."""

    def __init__(self) -> None:
        self.records: list[RequestMetrics] = []

    def __call__(self, metrics: RequestMetrics) -> None:
        self.records.append(metrics)


class _Provider(AIProvider):
    """Returns or raises a scripted sequence."""

    def __init__(self, outcomes: list[AIResponse | Exception]) -> None:
        self._outcomes = list(outcomes)

    @property
    def name(self) -> str:
        return "custom-provider"

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> AIResponse:
        del model, messages
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _guard(provider: _Provider, sink: _Sink, *, max_attempts: int = 3) -> AIGuard:
    return AIGuard(
        provider,
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_delay=0.0),
        sleep=lambda _delay: None,
        on_metrics=sink,
        clock=_Clock([1.0, 1.5]),
    )


def test_successful_metrics_copy_usage_and_cost_from_response() -> None:
    response = AIResponse(
        content="ok",
        model="test-model",
        input_tokens=12,
        output_tokens=8,
        cost=0.03,
    )
    sink = _Sink()
    guard = _guard(_Provider([response]), sink)

    result = guard.chat(model="test-model", messages=[{"role": "user", "content": "Hi"}])

    assert result is response
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.input_tokens == 12
    assert metrics.output_tokens == 8
    assert math.isclose(metrics.cost, 0.03)
    assert metrics.total_tokens == 20


def test_retry_success_records_only_final_response_usage() -> None:
    response = AIResponse(
        content="ok",
        model="test-model",
        input_tokens=9,
        output_tokens=4,
        cost=1.5,
    )
    sink = _Sink()
    guard = _guard(_Provider([RateLimitError("again"), response]), sink, max_attempts=2)

    result = guard.chat(model="test-model", messages=[{"role": "user", "content": "Hi"}])

    assert result is response
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.success is True
    assert metrics.attempts == 2
    assert metrics.input_tokens == 9
    assert metrics.output_tokens == 4
    assert math.isclose(metrics.cost, 1.5)


def test_failed_request_metrics_have_zero_usage_and_cost() -> None:
    error = AuthenticationError("denied")
    sink = _Sink()
    guard = _guard(_Provider([error]), sink)

    with pytest.raises(AuthenticationError) as caught:
        guard.chat(model="test-model", messages=[{"role": "user", "content": "Hi"}])

    assert caught.value is error
    assert len(sink.records) == 1
    metrics = sink.records[0]
    assert metrics.success is False
    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.cost == 0.0


def test_usage_api_imports_from_package() -> None:
    import ai_api_guard

    assert ai_api_guard.TokenUsage is TokenUsage
    assert ai_api_guard.ModelPricing is ModelPricing
    assert ai_api_guard.calculate_cost is calculate_cost
