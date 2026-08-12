"""Token counting, used for pre-call cost estimates.

Budgets are enforced before the provider call, which means we have to price a
request before anyone has told us how many tokens it contains. This module is
that estimate.

It uses one tokenizer for every provider, which is an approximation: Anthropic,
OpenAI, and Llama models tokenize the same text slightly differently. That
inaccuracy is real, so the savings report measures estimate-versus-actual error
and prints it rather than hiding it.
"""

from __future__ import annotations

from functools import lru_cache

#: Fallback ratio when no tokenizer is available. Close enough for English prose
#: and code; worse for other languages, which is part of the estimate error.
CHARS_PER_TOKEN = 4

#: Per-message framing overhead (role markers, separators) that every provider
#: adds around message content.
TOKENS_PER_MESSAGE = 4


@lru_cache
def _encoder():
    """Return a tiktoken encoder, or None if tiktoken isn't usable here.

    Cached because building an encoder reads a vocabulary file, and the estimate
    runs on every request.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - only hit when tiktoken is unavailable
        return None


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in `text`."""
    if not text:
        return 0

    encoder = _encoder()
    if encoder is None:
        return max(1, len(text) // CHARS_PER_TOKEN)
    return len(encoder.encode(text))


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate input tokens for a full message list, including framing overhead."""
    return sum(
        count_tokens(message.get("content", "")) + TOKENS_PER_MESSAGE
        for message in messages
    )
