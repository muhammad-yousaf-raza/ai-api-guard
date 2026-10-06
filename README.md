# AI API Guard

A provider-independent Python toolkit for adding reliability, observability, usage tracking, and cost awareness around AI API calls.

OpenAI is the first supported provider adapter. It covers synchronous chat completions. The library does not replace provider SDKs. It wraps a provider you supply.

[![PyPI](https://img.shields.io/pypi/v/ai-api-guard)](https://pypi.org/project/ai-api-guard/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/muhammad-yousaf-raza/ai-api-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammad-yousaf-raza/ai-api-guard/actions/workflows/ci.yml)

## Why AI API Guard?

Application code that calls an AI SDK directly usually owns the surrounding work: transient failures, retry behavior, request observability, token usage, cost calculation, and provider-specific exceptions.

AI API Guard is a small orchestration layer in front of a provider. Your code calls `AIGuard`. The guard calls an `AIProvider`, retries the transient failures it knows about, and can report one metrics record for the finished call.

## Features

- Provider-independent `AIProvider` abstraction
- `AIGuard` orchestration
- Configurable retries with exponential backoff
- Retries for transient provider failures
- Request metrics callback
- Latency measurement and attempt tracking
- Token usage tracking
- Caller-supplied pricing and cost calculation
- Provider-independent exception hierarchy
- Optional OpenAI adapter
- Typed, tested Python API

Not implemented yet:

- Async clients
- Streaming
- Automatic pricing downloads
- Built-in provider price lists

## Installation

Install the package from [PyPI](https://pypi.org/project/ai-api-guard/):

```bash
pip install ai-api-guard
```

For OpenAI support:

```bash
pip install "ai-api-guard[openai]"
```

The OpenAI extra installs the official OpenAI Python SDK. `OpenAIProvider()` does not take a hard-coded API key. When you omit `api_key` and `client`, the SDK reads `OPENAI_API_KEY` from the environment.

To install from a local checkout instead of PyPI:

```bash
git clone https://github.com/muhammad-yousaf-raza/ai-api-guard.git
cd ai-api-guard
python -m pip install -e ".[openai]"
```

Omit `[openai]` if you only need the provider-independent core. Development dependencies are covered under [Development](#development).

## Quick start

```python
from ai_api_guard import AIGuard
from ai_api_guard.providers.openai import OpenAIProvider

provider = OpenAIProvider()
guard = AIGuard(provider)

response = guard.chat(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain exponential backoff in one sentence.",
        }
    ],
)

print(response.content)
```

`examples/openai_basic.py` is the same kind of script. It also prints the model, token counts, and cost.

This example is documentation. It is not a record of a live API call.

Without a pricing map, `response.cost` stays `0.0`. See [Cost tracking](#cost-tracking).

## Retry configuration

`max_attempts` is the total number of provider calls, including the first attempt. It is not the number of extra retries.

```python
from ai_api_guard import AIGuard, RetryPolicy
from ai_api_guard.providers.openai import OpenAIProvider

policy = RetryPolicy(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=8.0,
    backoff_multiplier=2.0,
)

guard = AIGuard(OpenAIProvider(), retry_policy=policy)
```

These values are also the defaults. After a retryable failure, the guard waits `initial_delay` seconds, then multiplies that delay by `backoff_multiplier` for each later failure. The wait never exceeds `max_delay`.

`AIGuard` retries only:

- `RateLimitError`
- `ProviderTimeoutError`
- `ProviderUnavailableError`

The final failure is re-raised unchanged. `AuthenticationError`, a generic `ProviderError`, and any other exception are not retried.

## Observability

Pass `on_metrics` to receive one `RequestMetrics` record for each `chat` call. Intermediate retry attempts do not emit their own records.

```python
from ai_api_guard import AIGuard, RequestMetrics
from ai_api_guard.providers.openai import OpenAIProvider


def handle_metrics(metrics: RequestMetrics) -> None:
    print(metrics.provider)
    print(metrics.model)
    print(metrics.attempts)
    print(metrics.latency)
    print(metrics.total_tokens)
    print(metrics.cost)


guard = AIGuard(OpenAIProvider(), on_metrics=handle_metrics)
```

`latency` is the elapsed time for the whole guarded call, in seconds. `attempts` is the number of provider calls made. On success, token counts and `cost` are copied from the `AIResponse`. On failure they are zero, and `error_type` is the exception class name. `success` is `True` only when the call returns a response.

If `handle_metrics` raises, that exception is ignored. The chat result is unchanged.

## Cost tracking

Prices are not built in. You supply them. The numbers below are example values only, not current provider pricing, and they do not name a currency. Each number is a currency unit per 1,000,000 tokens.

```python
from ai_api_guard import ModelPricing
from ai_api_guard.providers.openai import OpenAIProvider

# Example values only — not current provider pricing.
pricing = {
    "example-model": ModelPricing(
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
}

provider = OpenAIProvider(pricing=pricing)
```

`OpenAIProvider` looks up the model name returned by the API first, then the model name you requested. When it finds a `ModelPricing` entry, it fills `response.cost`. When it does not, `response.cost` stays `0.0`.

## Exception handling

```python
from ai_api_guard import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

try:
    response = guard.chat(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
except AuthenticationError:
    print("credentials were rejected")
except (RateLimitError, ProviderTimeoutError, ProviderUnavailableError):
    print("retries were exhausted")
except ProviderError:
    print("the provider returned an error")
```

`guard` is the `AIGuard` constructed above. `RateLimitError`, `ProviderTimeoutError`, and `ProviderUnavailableError` are retried until `max_attempts` is exhausted, then raised. `AuthenticationError` and a generic `ProviderError` propagate on the first occurrence.

The OpenAI adapter translates SDK authentication, rate-limit, timeout, connection, and other API errors into these types. Other Python exceptions are left unchanged.

## Architecture

```text
Application
    |
    v
AIGuard
    |
    +-- RetryPolicy
    +-- RequestMetrics
    |
    v
AIProvider
    |
    +-- OpenAIProvider
    +-- Future Providers
    |
    v
AIResponse
```

`OpenAIProvider` is not imported by `import ai_api_guard`. Import it from `ai_api_guard.providers.openai` when the OpenAI extra is installed.

## Provider support

| Provider  | Status    |
| --------- | --------- |
| OpenAI    | Supported |
| Anthropic | Planned   |
| Gemini    | Planned   |

Planned providers are not implemented. OpenAI support is synchronous chat completions only.

## Development

```bash
python -m pip install -e ".[dev,openai]"
python -m pytest -v
python -m ruff check .
python -m mypy src
```

The test suite does not call the OpenAI API and does not need an API key.

## Project status

AI API Guard is pre-alpha (version 0.1.0). Public APIs may change before 1.0.

## Contributing

Bug reports, feature discussions, provider adapters, and improvements to tests or documentation are welcome. Open an issue or a pull request on [GitHub](https://github.com/muhammad-yousaf-raza/ai-api-guard).

## License

[MIT License](LICENSE)

## Author

Muhammad Yousaf Raza

- GitHub: [muhammad-yousaf-raza](https://github.com/muhammad-yousaf-raza)
- LinkedIn: [muhammad-yousaf-raza](https://www.linkedin.com/in/muhammad-yousaf-raza/)
