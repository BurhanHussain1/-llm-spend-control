"""A free, offline, deterministic provider.

This adapter is what makes the project demonstrable at zero cost: the test suite
and the 1,000-prompt workload simulation both run against it, so a reviewer can
clone the repo with no API keys and reproduce every number in the case study.

Two design choices make it useful rather than just cheap:

* **Deterministic.** The same prompt always produces the same answer, token
  counts, and latency, so the simulation's savings figure is reproducible.
* **Tier-aware.** Cheaper tiers return shorter, thinner answers. That gives the
  Phase 6 quality judge something real to disagree with, instead of every tier
  producing identical output and a meaningless 100% pass rate.

Reported latency is fabricated, not slept through -- a 1,000-request simulation
finishes in seconds rather than in an hour.
"""

from __future__ import annotations

import hashlib

from app.providers.base import Completion, ProviderError
from app.registry import ModelRegistry
from app.schemas import Message
from app.tokens import count_tokens

#: Output length as a multiple of the tier-1 baseline. A stronger model writes a
#: fuller answer, which is what the quality judge is meant to notice.
_WORDS_BY_TIER = {1: 25, 2: 60, 3: 120}

#: Prompts containing this marker return an empty answer, so tests can exercise
#: the degenerate-output path that triggers escalation in Phase 6.
DEGENERATE_MARKER = "[[mock:degenerate]]"


class MockProvider:
    """Fake completions with realistic token counts and latency."""

    name = "mock"

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    async def complete(
        self,
        model: str,
        messages: list[Message],
        max_output_tokens: int,
    ) -> Completion:
        spec = self._registry.get(model)
        if spec.provider != self.name:
            raise ProviderError(self.name, f"{model!r} is not a mock model")

        prompt = "\n".join(message.content for message in messages)
        digest = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

        if DEGENERATE_MARKER in prompt:
            text = ""
        else:
            text = self._answer(spec.tier, digest, prompt)

        input_tokens = count_tokens(prompt)
        output_tokens = min(count_tokens(text), max_output_tokens)

        return Completion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=self._latency_ms(spec.typical_latency_ms, digest),
            provider_metadata={"mock": True, "digest": digest[:12], "tier": spec.tier},
        )

    def _answer(self, tier: int, digest: str, prompt: str) -> str:
        """Build a deterministic answer whose depth scales with the tier."""
        target_words = _WORDS_BY_TIER[tier]
        opening = (
            f"[tier {tier} answer {digest[:8]}] "
            f"Responding to a {count_tokens(prompt)}-token prompt."
        )
        filler = " ".join(
            f"point-{index}" for index in range(1, max(1, target_words - 8) + 1)
        )
        return f"{opening} {filler}".strip()

    def _latency_ms(self, typical_ms: int, digest: str) -> int:
        """Jitter the model's typical latency deterministically, by +/- 20%."""
        spread = int(digest[:4], 16) % 41 - 20  # -20..+20
        return max(1, round(typical_ms * (1 + spread / 100)))
