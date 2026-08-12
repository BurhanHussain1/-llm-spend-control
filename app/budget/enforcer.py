"""Budget enforcement: estimate, reserve, settle.

The detail that makes this correct is an ordering problem. A budget has to be
enforced **before** the provider call, or the money is already spent -- but the
true cost is only known **after** the response, because nobody knows how many
tokens the model will emit. So:

1. **Estimate** the cost from the prompt (see ``estimator.py``).
2. **Reserve** it: add the estimate to the counters, then look at the new totals.
   If a limit broke, roll the increment back and reject before spending anything.
   Incrementing first and checking second is what makes this safe under
   concurrency -- read-then-write would let two requests each see room for one.
3. Call the provider.
4. **Settle**: move the counter by ``actual - estimate``, so what remains on the
   counter is the real cost. A failed call **releases** the whole reservation
   instead, which is why a provider outage never eats a team's budget.

The decision itself -- :func:`evaluate` -- is a pure function over totals and
limits, so every boundary can be tested without Redis or a clock.

This module touches counters only. Writing the ledger rows that make the trail
auditable is the gateway's job, so that a budget decision stays testable without
a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.budget.counters import SpendCounters
from app.budget.policies import BudgetPolicies, Limits
from app.db.models import utcnow
from app.db.repository import start_of_utc_day, start_of_utc_month
from app.schemas import BudgetStatus, Priority

#: Priorities that may proceed past a broken limit, with the override recorded.
OVERRIDE_PRIORITIES: tuple[Priority, ...] = ("high",)

#: Extra life given to a counter key beyond the end of its period, so a request
#: that lands on the boundary cannot read a key that has just been evicted.
TTL_GRACE_SECONDS = 3600


@dataclass(frozen=True)
class BudgetScope:
    """One limit a request is checked against."""

    key: str
    label: str
    """Human-readable, e.g. ``"daily budget for team 'search'"``. Goes in errors."""
    limit_usd: float
    ttl_seconds: int


@dataclass(frozen=True)
class BudgetDecision:
    """The outcome of a reservation attempt."""

    status: BudgetStatus
    reason: str
    percent_used: float
    """Highest utilisation across every scope checked, as a percentage."""
    reserved_usd: float
    """What is currently held on the counters. Zero on a block -- the increment
    was rolled back."""
    warnings: list[str] = field(default_factory=list)
    violated_scope: str | None = None

    @property
    def allowed(self) -> bool:
        return self.status in ("allow", "warn", "override")


def evaluate(
    totals: dict[str, float],
    scopes: list[BudgetScope],
    priority: Priority,
    warn_threshold: float,
    reserved_usd: float,
) -> BudgetDecision:
    """Decide whether a request may proceed, given the post-increment totals.

    Pure: no clock, no network, no database.

    A scope is broken only when spend goes **past** its limit, not when it lands
    exactly on it -- a team is entitled to spend its whole budget. The request
    after the one that consumes the last cent is the one that gets blocked.
    """
    if not scopes:
        raise ValueError("cannot evaluate a budget with no scopes")

    utilisation = {
        scope.key: totals.get(scope.key, 0.0) / scope.limit_usd for scope in scopes
    }
    percent_used = round(max(utilisation.values()) * 100, 2)

    broken = [scope for scope in scopes if utilisation[scope.key] > 1.0]

    if broken:
        worst = max(broken, key=lambda scope: utilisation[scope.key])
        spent = totals.get(worst.key, 0.0)

        if priority in OVERRIDE_PRIORITIES:
            return BudgetDecision(
                status="override",
                reason=(
                    f"{worst.label} exceeded (${spent:.4f} of ${worst.limit_usd:.2f}); "
                    f"allowed because priority is {priority!r}"
                ),
                percent_used=percent_used,
                reserved_usd=reserved_usd,
                warnings=[f"budget override: {worst.label} is over its limit"],
                violated_scope=worst.key,
            )

        return BudgetDecision(
            status="blocked",
            reason=(
                f"{worst.label} exhausted: this request would take spend to "
                f"${spent:.4f} against a limit of ${worst.limit_usd:.2f}"
            ),
            percent_used=percent_used,
            reserved_usd=0.0,
            violated_scope=worst.key,
        )

    warnings = [
        f"{scope.label} is at {utilisation[scope.key] * 100:.1f}% of its limit"
        for scope in scopes
        if utilisation[scope.key] >= warn_threshold
    ]
    if warnings:
        return BudgetDecision(
            status="warn",
            reason="within budget, but past the warning threshold",
            percent_used=percent_used,
            reserved_usd=reserved_usd,
            warnings=warnings,
        )

    return BudgetDecision(
        status="allow",
        reason="within budget",
        percent_used=percent_used,
        reserved_usd=reserved_usd,
    )


class BudgetEnforcer:
    """Applies budget policy to counters."""

    def __init__(
        self,
        policies: BudgetPolicies,
        counters: SpendCounters,
        warn_threshold: float = 0.8,
    ) -> None:
        self._policies = policies
        self._counters = counters
        self._warn_threshold = warn_threshold

    # --- scopes --------------------------------------------------------------

    def scopes_for(
        self,
        team_id: str,
        feature: str,
        now: datetime | None = None,
    ) -> list[BudgetScope]:
        """Every limit this request must satisfy, team-level plus feature-level."""
        now = now or utcnow()
        team_limits = self._policies.for_team(team_id)

        scopes = [
            _day_scope(f"spend:{team_id}", f"daily budget for team {team_id!r}",
                       team_limits.daily_limit_usd, now),
            _month_scope(f"spend:{team_id}", f"monthly budget for team {team_id!r}",
                         team_limits.monthly_limit_usd, now),
        ]

        feature_limits = self._policies.for_feature(team_id, feature)
        if feature_limits is not None:
            prefix = f"spend:{team_id}:{feature}"
            label = f"{team_id}/{feature}"
            scopes.append(
                _day_scope(prefix, f"daily budget for {label}",
                           feature_limits.daily_limit_usd, now)
            )
            scopes.append(
                _month_scope(prefix, f"monthly budget for {label}",
                             feature_limits.monthly_limit_usd, now)
            )
        return scopes

    # --- the three steps -----------------------------------------------------

    async def reserve(
        self,
        team_id: str,
        feature: str,
        priority: Priority,
        estimated_cost_usd: float,
        now: datetime | None = None,
    ) -> BudgetDecision:
        """Hold `estimated_cost_usd` against the budget, or refuse the request.

        Increments first and inspects the result, so two concurrent requests can
        never both pass a check that only had room for one. A refusal rolls its
        own increment back before returning.
        """
        if estimated_cost_usd < 0:
            raise ValueError("estimated cost cannot be negative")

        scopes = self.scopes_for(team_id, feature, now)
        keys = [(scope.key, scope.ttl_seconds) for scope in scopes]

        totals = await self._counters.add(keys, estimated_cost_usd)
        decision = evaluate(
            totals=totals,
            scopes=scopes,
            priority=priority,
            warn_threshold=self._warn_threshold,
            reserved_usd=estimated_cost_usd,
        )

        if decision.status == "blocked":
            await self._counters.add(keys, -estimated_cost_usd)

        return decision

    async def settle(
        self,
        team_id: str,
        feature: str,
        reserved_usd: float,
        actual_cost_usd: float,
        now: datetime | None = None,
    ) -> float:
        """Correct the counters from the estimate to the real cost.

        Returns the applied delta: negative when the estimate was high (budget
        handed back), positive when it was low.
        """
        delta = round(actual_cost_usd - reserved_usd, 8)
        if delta:
            await self._adjust(team_id, feature, delta, now)
        return delta

    async def release(
        self,
        team_id: str,
        feature: str,
        reserved_usd: float,
        now: datetime | None = None,
    ) -> None:
        """Hand back a whole reservation, because the call never produced output."""
        if reserved_usd:
            await self._adjust(team_id, feature, -reserved_usd, now)

    async def _adjust(
        self, team_id: str, feature: str, delta: float, now: datetime | None
    ) -> None:
        scopes = self.scopes_for(team_id, feature, now)
        await self._counters.add(
            [(scope.key, scope.ttl_seconds) for scope in scopes], delta
        )

    # --- reporting and startup ----------------------------------------------

    async def snapshot(
        self,
        team_id: str,
        feature: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, float]:
        """Current spend and limits for a team, for ``GET /v1/budgets/{team_id}``."""
        now = now or utcnow()
        limits = self._policies.for_team(team_id)
        day = _day_scope(f"spend:{team_id}", "", limits.daily_limit_usd, now)
        month = _month_scope(f"spend:{team_id}", "", limits.monthly_limit_usd, now)

        totals = await self._counters.get([day.key, month.key])
        daily_spend = totals.get(day.key, 0.0)
        monthly_spend = totals.get(month.key, 0.0)

        return {
            "daily_spend_usd": round(daily_spend, 8),
            "daily_limit_usd": limits.daily_limit_usd,
            "monthly_spend_usd": round(monthly_spend, 8),
            "monthly_limit_usd": limits.monthly_limit_usd,
            "daily_percent_used": round(daily_spend / limits.daily_limit_usd * 100, 2),
            "monthly_percent_used": round(
                monthly_spend / limits.monthly_limit_usd * 100, 2
            ),
        }

    async def seed(
        self,
        team_id: str,
        daily_spend_usd: float,
        monthly_spend_usd: float,
        now: datetime | None = None,
    ) -> None:
        """Set a team's counters to known spend, overwriting whatever is there.

        Called on startup from the usage log. Without this, restarting a gateway
        that uses in-memory counters would reset every team's counters to zero
        while the database still shows the day's real spend -- handing out a
        fresh budget on every deploy.
        """
        now = now or utcnow()
        limits = self._policies.for_team(team_id)
        day = _day_scope(f"spend:{team_id}", "", limits.daily_limit_usd, now)
        month = _month_scope(f"spend:{team_id}", "", limits.monthly_limit_usd, now)

        await self._counters.set(day.key, daily_spend_usd, day.ttl_seconds)
        await self._counters.set(month.key, monthly_spend_usd, month.ttl_seconds)


def _day_scope(prefix: str, label: str, limit: float, now: datetime) -> BudgetScope:
    start = start_of_utc_day(now)
    end = start + timedelta(days=1)
    return BudgetScope(
        key=f"{prefix}:day:{start:%Y-%m-%d}",
        label=label,
        limit_usd=limit,
        ttl_seconds=_seconds_until(end, now),
    )


def _month_scope(prefix: str, label: str, limit: float, now: datetime) -> BudgetScope:
    start = start_of_utc_month(now)
    # Jump into the next month, then snap back to its first day.
    end = start_of_utc_month(start + timedelta(days=32))
    return BudgetScope(
        key=f"{prefix}:month:{start:%Y-%m}",
        label=label,
        limit_usd=limit,
        ttl_seconds=_seconds_until(end, now),
    )


def _seconds_until(moment: datetime, now: datetime) -> int:
    """Seconds from `now` to `moment`, plus grace, floored at the grace period."""
    remaining = (moment - now.astimezone(timezone.utc)).total_seconds()
    return max(int(remaining), 0) + TTL_GRACE_SECONDS
