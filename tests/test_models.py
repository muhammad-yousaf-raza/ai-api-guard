"""Tests for provider-independent response models."""

from ai_api_guard.models import AIResponse


def test_ai_response_stores_content() -> None:
    response = AIResponse(content="hello", model="test-model")

    assert response.content == "hello"


def test_total_tokens_adds_input_and_output() -> None:
    response = AIResponse(
        content="hello",
        model="test-model",
        input_tokens=3,
        output_tokens=5,
    )

    assert response.total_tokens == 8


def test_default_input_tokens_is_zero() -> None:
    response = AIResponse(content="hello", model="test-model")

    assert response.input_tokens == 0


def test_default_output_tokens_is_zero() -> None:
    response = AIResponse(content="hello", model="test-model")

    assert response.output_tokens == 0


def test_default_cost_is_zero() -> None:
    response = AIResponse(content="hello", model="test-model")

    assert response.cost == 0.0


def test_default_latency_is_zero() -> None:
    response = AIResponse(content="hello", model="test-model")

    assert response.latency == 0.0


def test_default_raw_response_is_none() -> None:
    response = AIResponse(content="hello", model="test-model")

    assert response.raw_response is None
