"""The contract every provider adapter satisfies.

An adapter's only job is to turn one vendor's API into a :class:`Completion`.
Nothing else in the codebase knows or cares which vendor answered a request --
that is what makes the router free to send the same prompt anywhere.

Adapters do not compute cost. Cost is registry arithmetic over the token counts
they return, so pricing lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.schemas import Message


class ProviderError(RuntimeError):
    """A provider call failed.

    The gateway maps this to a 503 and releases the request's budget
    reservation, so a provider outage never eats a team's budget.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


@dataclass(frozen=True)
class Completion:
    """One normalized provider response."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    """Raw vendor fields worth keeping (stop reason, request id) for debugging."""


@runtime_checkable
class Provider(Protocol):
    """What the router needs from an adapter."""

    name: str

    async def complete(
        self,
        model: str,
        messages: list[Message],
        max_output_tokens: int,
    ) -> Completion:
        """Run one completion, or raise :class:`ProviderError`."""
        ...


def split_system_prompt(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Separate leading system turns from the conversation.

    Anthropic takes the system prompt as its own top-level parameter rather than
    a message, so adapters that need it call this instead of each re-deriving it.
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    conversation = [m for m in messages if m.role != "system"]
    system = "\n\n".join(system_parts) if system_parts else None
    return system, conversation
