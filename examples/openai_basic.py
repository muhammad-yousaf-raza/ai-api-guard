"""One synchronous OpenAI chat call through AIGuard.

Requires the openai extra and OPENAI_API_KEY in the environment.
The OpenAI SDK reads that variable when OpenAIProvider is constructed
without an api_key. Cost stays 0.0 unless caller-supplied pricing is
configured on the provider.
"""

from ai_api_guard import AIGuard
from ai_api_guard.providers.openai import OpenAIProvider


def main() -> None:
    """Send one chat request and print the normalized response."""
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
    # Cost requires caller-supplied pricing. It stays 0.0 in this example.
    print(response.content)
    print(f"model: {response.model}")
    print(f"input tokens: {response.input_tokens}")
    print(f"output tokens: {response.output_tokens}")
    print(f"total tokens: {response.total_tokens}")
    print(f"cost: {response.cost}")


if __name__ == "__main__":
    main()
