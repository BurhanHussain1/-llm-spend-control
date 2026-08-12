"""Ollama adapter for locally hosted models, via plain `httpx`.

Ollama runs on the machine, so these calls carry no per-token bill -- which is
what makes them the cheapest possible tier-1 destination. Because it is local,
a connection error usually means "the daemon isn't running" rather than a
provider outage, so the error message says so.
"""

from __future__ import annotations

import time

import httpx

from app.providers.base import Completion, ProviderError
from app.schemas import Message


class OllamaProvider:
    """Calls a local Ollama server's `/api/chat` endpoint."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def complete(
        self,
        model: str,
        messages: list[Message],
        max_output_tokens: int,
    ) -> Completion:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "options": {"num_predict": max_output_tokens},
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat", json=payload
                )
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderError(
                self.name,
                f"cannot reach Ollama at {self._base_url} -- is `ollama serve` running?",
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, str(exc)) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)

        body = response.json()

        return Completion(
            text=body.get("message", {}).get("content", ""),
            # Ollama omits these counts when a response is served from its cache.
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            latency_ms=latency_ms,
            provider_metadata={
                "model": body.get("model"),
                "done_reason": body.get("done_reason"),
            },
        )
