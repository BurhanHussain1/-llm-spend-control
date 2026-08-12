"""Anthropic adapter, via the official `anthropic` SDK.

Two Anthropic-specific details this handles so nothing else has to:

* the system prompt is a top-level parameter, not a message
* a response can come back with ``stop_reason == "refusal"``, which is a
  successful HTTP call with no usable content -- so `stop_reason` is checked
  before the content blocks are read

We deliberately do **not** set the `thinking` parameter. This service exists to
control spend, and thinking tokens are billed, so the caller's cost profile is
left as the model's own default rather than inflated by the gateway. Note that
`max_tokens` caps thinking *plus* visible text on current models, which is why
the request default is generous.
"""

from __future__ import annotations

import time

from anthropic import APIError, AsyncAnthropic

from app.providers.base import Completion, ProviderError, split_system_prompt
from app.schemas import Message


class AnthropicProvider:
    """Calls the Anthropic Messages API and normalizes the result."""

    name = "anthropic"

    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)

    async def complete(
        self,
        model: str,
        messages: list[Message],
        max_output_tokens: int,
    ) -> Completion:
        system, conversation = split_system_prompt(messages)

        request: dict[str, object] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in conversation
            ],
        }
        if system:
            request["system"] = system

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(**request)  # type: ignore[arg-type]
        except APIError as exc:
            raise ProviderError(self.name, str(exc)) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        # Check why generation stopped before trusting the content blocks: on a
        # refusal the call succeeds but `content` is empty or partial.
        refused = response.stop_reason == "refusal"
        text = "" if refused else _extract_text(response)

        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            provider_metadata={
                "id": response.id,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "refused": refused,
            },
        )


def _extract_text(response: object) -> str:
    """Concatenate the text blocks of a Messages API response.

    A response can also carry thinking or tool-use blocks, so blocks are
    filtered by type rather than indexed positionally.
    """
    blocks = getattr(response, "content", []) or []
    return "".join(
        block.text for block in blocks if getattr(block, "type", None) == "text"
    )
