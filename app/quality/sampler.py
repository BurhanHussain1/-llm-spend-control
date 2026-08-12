"""Deciding which responses get verified, and which get escalated.

Two separate decisions live here, both cheap and both pure:

* :func:`should_shadow_check` -- sample a fraction of cheap-model answers for
  offline comparison against a stronger model. This is what produces a measured
  quality number to sit beside the savings number.
* :func:`should_escalate` -- rerun *now*, before answering, because the cheap
  answer looks unusable.

Sampling is deterministic on the prompt rather than random. The same workload
therefore verifies the same requests on every run, which is what makes the
simulation's pass rate reproducible instead of drifting between runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.providers.base import Completion
from app.registry import Model
from app.schemas import ChatRequest

#: An answer shorter than this is treated as degenerate rather than terse.
MIN_USEFUL_OUTPUT_TOKENS = 3

#: Sampling resolution. 10,000 buckets makes rates as fine as 0.01% expressible.
_BUCKETS = 10_000


def should_shadow_check(
    prompt: str,
    model: Model,
    strongest_tier: int,
    rate: float,
) -> bool:
    """True if this response should be compared against a stronger model.

    Only cheaper models are sampled: verifying the strongest model against itself
    would cost money and prove nothing.
    """
    if rate <= 0 or model.tier >= strongest_tier:
        return False
    if rate >= 1:
        return True

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % _BUCKETS
    return bucket < rate * _BUCKETS


@dataclass(frozen=True)
class EscalationDecision:
    """Whether to rerun, why, and which kind of reason it was."""

    needed: bool
    reason: str = ""
    category: str = ""
    """``degenerate`` -- the answer is unusable, so correctness demands a rerun.
    ``priority`` -- the answer looks fine but the request was urgent.

    The gateway treats these differently when a budget is already exhausted: a
    broken answer is still worth fixing, but spending twice on a *precaution*
    past a blown budget is not."""


def should_escalate(
    request: ChatRequest,
    completion: Completion,
    model: Model,
    strongest_tier: int,
    escalate_high_priority: bool,
) -> EscalationDecision:
    """Decide whether to discard this answer and rerun on a stronger model.

    The reason is recorded on the rerun's usage row, so an escalation is never a
    mystery after the fact.

    Note what is *not* here: risk tags. Those are handled at routing time, where a
    tagged request goes straight to a strong model instead of being answered
    cheaply and then done twice.
    """
    if model.tier >= strongest_tier:
        return EscalationDecision(needed=False)

    text = completion.text.strip()
    if not text:
        return EscalationDecision(
            True, "primary model returned an empty answer", "degenerate"
        )
    if completion.output_tokens < MIN_USEFUL_OUTPUT_TOKENS:
        return EscalationDecision(
            True,
            f"primary answer was {completion.output_tokens} tokens, "
            "which is too short to be a real answer",
            "degenerate",
        )
    if completion.provider_metadata.get("refused"):
        return EscalationDecision(
            True, "primary model declined the request", "degenerate"
        )

    if escalate_high_priority and request.priority == "high" and model.tier == 1:
        # Deliberately limited to tier 1, where the quality gap is widest.
        # Rerunning every high-priority request would double its cost, which is
        # the reflex this whole service exists to question -- so the escalation
        # cost is reported separately rather than folded into routed spend.
        return EscalationDecision(
            True, "high-priority request was routed to a tier-1 model", "priority"
        )

    return EscalationDecision(needed=False)
