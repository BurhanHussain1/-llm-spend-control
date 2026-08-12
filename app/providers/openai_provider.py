"""OpenAI adapter, via the official `openai` SDK.

OpenAI takes the system prompt as an ordinary message, so this adapter passes
the conversation straight through. It uses `max_completion_tokens` rather than
the deprecated `max_tokens`.
"""

from __future__ import annotations

import time

from openai import APIError, AsyncOpenAI

from app.providers.base import Completion, ProviderError
from app.schemas import Message


class OpenAIProvider:
    """Calls the OpenAI Chat Completions API and normalizes the result."""

    name = "openai"

    def __init__(self, api_key: str, timeout_seconds: float = 30.0) -> None:
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    async def complete(
        self,
        model: str,
        messages: list[Message],
        max_output_tokens: int,
    ) -> Completion:
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=model,
                max_completion_tokens=max_output_tokens,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],  # type: ignore[arg-type]
            )
        except APIError as exc:
            raise ProviderError(self.name, str(exc)) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        choice = response.choices[0]
        usage = response.usage

        return Completion(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            provider_metadata={
                "id": response.id,
                "model": response.model,
                "finish_reason": choice.finish_reason,
            },
        )
