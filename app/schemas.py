"""The gateway's public request and response shapes.

Callers send one :class:`ChatRequest` regardless of which provider ends up
serving it, and always get one :class:`ChatResponse` back. Normalizing here is
what lets the router swap models freely without breaking any caller.

The response deliberately reports *why* it did what it did -- `routing_reason`,
`budget_status`, `baseline_cost_usd` -- so a single curl is enough to understand
a decision without reading the logs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["low", "normal", "high"]
Role = Literal["system", "user", "assistant"]

#: Returned in `budget_status`. WARN and ALLOW both mean the call went through.
BudgetStatus = Literal["allow", "warn", "blocked", "override"]


class Message(BaseModel):
    """One chat turn, in the shape every provider adapter accepts."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """A chat completion request, plus the metadata budgets and routing need."""

    messages: list[Message] = Field(min_length=1)

    # --- who is spending -----------------------------------------------------
    team_id: str = Field(min_length=1, description="Billing owner. Budgets apply here.")
    feature: str = Field(min_length=1, description="Product surface making the call.")

    # --- how the request should be handled -----------------------------------
    priority: Priority = "normal"
    """`low` is blocked first when a budget runs out; `high` can override it."""

    preferred_model: str | None = None
    """Skips the classifier. Still subject to budgets and capability checks."""

    risk_tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags (e.g. 'legal', 'customer-facing') that push "
        "a request up a tier and make it eligible for auto-escalation.",
    )

    max_output_tokens: int = Field(default=4096, gt=0, le=128_000)
    """A ceiling, not a reservation -- you are billed for tokens actually produced.

    The default is deliberately generous because on current Anthropic models
    this cap covers thinking tokens as well as visible text, and a tight cap
    truncates the answer mid-sentence.
    """

    def prompt_text(self) -> str:
        """All message content as one string, for token counting and classifying."""
        return "\n".join(message.content for message in self.messages)


class ChatResponse(BaseModel):
    """The answer, plus a full account of how it was produced and what it cost."""

    text: str

    # --- what ran ------------------------------------------------------------
    chosen_model: str
    provider: str
    tier: int
    routing_reason: str
    """Human-readable: why this model and not another."""

    # --- what it cost --------------------------------------------------------
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    """What the same call would have cost on the registry's baseline model."""

    latency_ms: int

    # --- budget outcome ------------------------------------------------------
    budget_status: BudgetStatus
    warnings: list[str] = Field(default_factory=list)

    escalated: bool = False
    """True when a first, cheaper attempt was discarded and rerun on a stronger model."""

    @property
    def savings_usd(self) -> float:
        """How much routing saved on this request versus the baseline model."""
        return round(self.baseline_cost_usd - self.cost_usd, 8)


class BudgetView(BaseModel):
    """Current spend against limits, for `GET /v1/budgets/{team_id}`."""

    team_id: str
    daily_spend_usd: float
    daily_limit_usd: float
    monthly_spend_usd: float
    monthly_limit_usd: float
    daily_percent_used: float
    monthly_percent_used: float
    status: BudgetStatus


class ErrorDetail(BaseModel):
    """Every non-2xx response body. Says what happened and what to do next."""

    error: str
    message: str
    remedy: str | None = None
