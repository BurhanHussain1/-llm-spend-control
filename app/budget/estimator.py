"""Pre-call cost estimation.

The budget has to be checked before the provider is called, which means pricing a
request before its output exists. Two guesses are involved:

* **Input tokens** come from a tokenizer, so they are close but not exact -- one
  tokenizer stands in for three providers (see ``app/tokens.py``).
* **Output tokens** are genuinely unknown. The heuristic here is a fraction of
  the input length, capped by the caller's ``max_output_tokens``.

Both guesses are wrong by some amount, and that amount matters: if the estimate
is far from the actual cost, then enforcing a budget against the estimate is
theatre. So every estimate is stored alongside the real cost in
``usage_events``, and the savings report prints the error rather than hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.registry import Model
from app.schemas import Message
from app.tokens import count_message_tokens

#: Assumed output length as a fraction of input length. Chat and summarization
#: answers are usually shorter than their prompts; a request whose real output
#: runs longer than this simply settles upward after the call.
DEFAULT_OUTPUT_RATIO = 0.5

#: Floor on assumed output, so a one-line prompt still reserves something.
MIN_ESTIMATED_OUTPUT_TOKENS = 64


@dataclass(frozen=True)
class Estimate:
    """A priced guess at what a request will cost."""

    input_tokens: int
    output_tokens: int
    cost_usd: float

    def error_against(self, actual_cost_usd: float) -> float:
        """Signed error as a fraction of the actual cost.

        Positive means the estimate was high (budget was over-reserved and handed
        back on settle); negative means it was low.
        """
        if actual_cost_usd <= 0:
            return 0.0
        return round((self.cost_usd - actual_cost_usd) / actual_cost_usd, 4)


def estimate_cost(
    messages: list[Message],
    model: Model,
    max_output_tokens: int,
    output_ratio: float = DEFAULT_OUTPUT_RATIO,
) -> Estimate:
    """Price a request before it runs, on `model`."""
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")

    input_tokens = count_message_tokens(
        [{"role": message.role, "content": message.content} for message in messages]
    )

    assumed_output = max(
        MIN_ESTIMATED_OUTPUT_TOKENS, int(input_tokens * output_ratio)
    )
    output_tokens = min(assumed_output, max_output_tokens)

    return Estimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=model.cost(input_tokens, output_tokens),
    )
