"""Database tables.

Three tables carry the whole project's evidence:

* ``usage_events`` -- one row per request attempt. Every number the dashboard
  and the case study report is an aggregate over this table.
* ``budget_ledger`` -- the reserve/settle/release trail, which is what makes
  "we enforced the budget before the call" auditable rather than a claim.
* ``routing_misses`` -- requests where the cheap model's answer failed the
  quality check. This is the honesty check on the savings number.

Column types are chosen for portability: the same definitions run on SQLite
locally and Postgres under Docker Compose with no changes. Money is stored as
``Float``, which is fine at this scale; a production system billing customers
would use ``Numeric`` to avoid accumulating rounding drift.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware current time.

    Budget periods roll over on UTC day and month boundaries, so every timestamp
    is stored in UTC and the dashboard converts for display.
    """
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class UsageEvent(Base):
    """One request attempt: what ran, what it cost, and how it ended.

    Failed and blocked attempts are rows too, with ``cost_usd = 0``. Without
    them there is no way to report a block rate or an error rate by provider, so
    this table records attempts rather than only successes. Spend queries filter
    on ``status == 'ok'``.

    An escalated request writes **two** rows -- the cheap attempt and the
    stronger rerun -- because both were paid for. Escalation is not free and the
    report should say so.
    """

    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # --- who is spending -----------------------------------------------------
    team_id: Mapped[str] = mapped_column(String(64), nullable=False)
    feature: Mapped[str] = mapped_column(String(64), nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False)

    # --- what ran ------------------------------------------------------------
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[int] = mapped_column(Integer, nullable=False)
    routing_reason: Mapped[str] = mapped_column(String(255), nullable=False)

    # --- outcome -------------------------------------------------------------
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    """``ok`` | ``blocked`` | ``provider_error``."""

    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # --- money ---------------------------------------------------------------
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    """What the pre-call estimate predicted. Compared against ``cost_usd`` to
    report estimate error -- if the estimate is far off, budget enforcement is
    theatre, and that needs to be visible rather than hidden."""

    baseline_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    """What this call would have cost on the registry's baseline model. Stored
    per row so savings are a query over data, not a spreadsheet afterwards."""

    escalated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    """True on the stronger rerun of a request that was escalated."""

    prompt_preview: Mapped[str] = mapped_column(Text, default="", nullable=False)
    """First few hundred characters of the prompt, for the "most expensive
    prompts" view. Prompt text can contain user data -- see the privacy note in
    the README's known gaps."""

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Covers the hot path: "what has this team spent since <timestamp>".
        Index("ix_usage_team_created", "team_id", "created_at"),
        Index("ix_usage_feature_created", "feature", "created_at"),
        Index("ix_usage_model", "model"),
    )


class BudgetLedgerEntry(Base):
    """One movement of a team's budget counter.

    A successful request produces two entries: a ``reserve`` for the estimate
    before the call, and a ``settle`` for the real cost after it. A failed call
    produces a ``reserve`` and a ``release``, which is the paper trail proving a
    provider outage did not consume anyone's budget.
    """

    __tablename__ = "budget_ledger"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    team_id: Mapped[str] = mapped_column(String(64), nullable=False)
    feature: Mapped[str] = mapped_column(String(64), nullable=False)

    action: Mapped[str] = mapped_column(String(16), nullable=False)
    """``reserve`` | ``settle`` | ``release`` | ``override``."""

    amount_usd: Mapped[float] = mapped_column(Float, nullable=False)
    """Signed: a reserve is positive, a release is negative, a settle is the
    correction that takes the counter from the estimate to the actual cost."""

    usage_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("usage_events.id"), nullable=True
    )
    note: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    __table_args__ = (Index("ix_ledger_team_created", "team_id", "created_at"),)


class RoutingMiss(Base):
    """A request the router sent to a model that turned out not to be good enough.

    Written when the shadow quality check fails: the cheap answer did not hold
    up against the strong model's. This table is what turns "we saved 68%" into
    a claim with a measured quality cost attached.
    """

    __tablename__ = "routing_misses"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    usage_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("usage_events.id"), nullable=True
    )

    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    chosen_model: Mapped[str] = mapped_column(String(64), nullable=False)
    better_model: Mapped[str] = mapped_column(String(64), nullable=False)

    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    """False for a genuine miss. Passing checks are stored too, because the
    verifier pass rate needs a denominator."""

    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)

    __table_args__ = (Index("ix_miss_created", "created_at"),)
