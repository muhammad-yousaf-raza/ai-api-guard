# Changelog

## [0.2.0]

### Added

- Provider-independent Retry-After support on retryable provider errors.
- Retry-After parsing in the optional OpenAI adapter, without attaching SDK response objects to core errors.
- Configurable equal jitter on `RetryPolicy`. Jitter is off by default, so existing deterministic delays stay unchanged unless you enable it.
- `RetryEvent` and an optional `AIGuard` `on_retry` callback. The event reports the actual selected wait for a retry that will happen.
- A `py.typed` marker so downstream type checkers can see the public annotations.

`RequestMetrics` is still the single final record for each guarded call.

## [0.1.0]

Initial public pre-alpha release.

### Added

- Provider-independent `AIProvider` abstraction and `AIGuard` orchestration.
- Configurable exponential retry policy.
- Request metrics, token usage, and caller-supplied cost helpers.
- Optional OpenAI chat adapter.
- Continuous integration and package metadata for a PyPI release.
