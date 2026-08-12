"""Every database read and write in the project lives here.

Nothing else in the codebase issues SQL. The API, the budget enforcer, the
dashboard, and the simulation all call these methods, which means the schema can
change without touching seven files, and every query is testable in one place.

Phase 7 adds the reporting aggregates to this file. Derived figures that are not
SQL -- percentiles, month-end projections -- belong in ``app/reporting.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import BudgetLedgerEntry, RoutingMiss, UsageEvent, utcnow

#: Only these rows represent money actually spent. Blocked and errored attempts
#: are recorded for auditing but must never inflate a spend total.
BILLABLE_STATUS = "ok"

PROMPT_PREVIEW_CHARS = 500

#: Decimal places kept on summed dollar amounts. Matches the registry so a
#: single request's cost and a team's daily total are rounded the same way.
COST_PRECISION = 8


def start_of_utc_day(moment: datetime | None = None) -> datetime:
    """Midnight UTC on the day of `moment`. Daily budgets reset here."""
    moment = moment or utcnow()
    return moment.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def start_of_utc_month(moment: datetime | None = None) -> datetime:
    """Midnight UTC on the first of the month. Monthly budgets reset here."""
    return start_of_utc_day(moment).replace(day=1)


class Repository:
    """Data access for one unit of work.

    Callers own the transaction: methods stage changes and flush when they need
    a generated id, but never commit. That keeps a request's usage row and its
    ledger entries in a single atomic write.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # --- writes --------------------------------------------------------------

    def log_usage(
        self,
        *,
        team_id: str,
        feature: str,
        priority: str,
        model: str,
        provider: str,
        tier: int,
        routing_reason: str,
        status: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        estimated_cost_usd: float = 0.0,
        baseline_cost_usd: float = 0.0,
        kind: str = "primary",
        prompt: str = "",
        error: str | None = None,
        created_at: datetime | None = None,
    ) -> UsageEvent:
        """Record one request attempt and return it with its id populated.

        `created_at` defaults to now. It is settable so the workload simulation
        can spread its requests across a date range, giving the dashboard's daily
        chart more than one bar.
        """
        event = UsageEvent(
            team_id=team_id,
            feature=feature,
            priority=priority,
            model=model,
            provider=provider,
            tier=tier,
            routing_reason=routing_reason,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            estimated_cost_usd=estimated_cost_usd,
            baseline_cost_usd=baseline_cost_usd,
            kind=kind,
            prompt_preview=prompt[:PROMPT_PREVIEW_CHARS],
            error=error,
        )
        if created_at is not None:
            event.created_at = created_at
        self.session.add(event)
        self.session.flush()  # assigns event.id for the ledger to reference
        return event

    def record_ledger(
        self,
        *,
        team_id: str,
        feature: str,
        action: str,
        amount_usd: float,
        usage_event_id: int | None = None,
        note: str = "",
    ) -> BudgetLedgerEntry:
        """Record one budget movement (reserve, settle, release, or override)."""
        entry = BudgetLedgerEntry(
            team_id=team_id,
            feature=feature,
            action=action,
            amount_usd=amount_usd,
            usage_event_id=usage_event_id,
            note=note,
        )
        self.session.add(entry)
        return entry

    def record_quality_check(
        self,
        *,
        prompt: str,
        chosen_model: str,
        better_model: str,
        passed: bool | None,
        score: float | None = None,
        reason: str = "",
        usage_event_id: int | None = None,
    ) -> RoutingMiss:
        """Record a shadow quality check.

        Passing checks are stored as well as failures -- the verifier pass rate
        needs a denominator, and a table of only failures cannot provide one.
        Pass ``passed=None`` when the check could not be run; those rows are
        excluded from the pass rate rather than counted against it.
        """
        check = RoutingMiss(
            prompt=prompt,
            chosen_model=chosen_model,
            better_model=better_model,
            passed=passed,
            score=score,
            reason=reason,
            usage_event_id=usage_event_id,
        )
        self.session.add(check)
        return check

    # --- spend, for budget enforcement --------------------------------------

    def spend_since(
        self,
        team_id: str,
        since: datetime,
        feature: str | None = None,
    ) -> float:
        """Total billable spend for a team (optionally one feature) since `since`."""
        query = select(func.coalesce(func.sum(UsageEvent.cost_usd), 0.0)).where(
            UsageEvent.team_id == team_id,
            UsageEvent.created_at >= since,
            UsageEvent.status == BILLABLE_STATUS,
        )
        if feature is not None:
            query = query.where(UsageEvent.feature == feature)
        return _money(self.session.execute(query).scalar_one())

    def spend_today(self, team_id: str, feature: str | None = None) -> float:
        return self.spend_since(team_id, start_of_utc_day(), feature)

    def spend_this_month(self, team_id: str, feature: str | None = None) -> float:
        return self.spend_since(team_id, start_of_utc_month(), feature)

    # --- totals, for the savings report -------------------------------------

    def totals(self, since: datetime | None = None) -> dict[str, float]:
        """Aggregate actual spend against the baseline-model counterfactual.

        This single query is where the headline savings percentage comes from.
        """
        query = select(
            func.count(UsageEvent.id),
            func.coalesce(func.sum(UsageEvent.cost_usd), 0.0),
            func.coalesce(func.sum(UsageEvent.baseline_cost_usd), 0.0),
            func.coalesce(func.sum(UsageEvent.estimated_cost_usd), 0.0),
        ).where(UsageEvent.status == BILLABLE_STATUS)
        if since is not None:
            query = query.where(UsageEvent.created_at >= since)

        requests, actual, baseline, estimated = self.session.execute(query).one()

        return {
            "requests": int(requests),
            "actual_cost_usd": _money(actual),
            "baseline_cost_usd": _money(baseline),
            "estimated_cost_usd": _money(estimated),
            "savings_usd": _money(float(baseline) - float(actual)),
            "savings_percent": _percent_saved(float(baseline), float(actual)),
        }

    def count_by_status(self, since: datetime | None = None) -> dict[str, int]:
        """Attempts grouped by outcome -- the source of block and error rates."""
        query = select(UsageEvent.status, func.count(UsageEvent.id)).group_by(
            UsageEvent.status
        )
        if since is not None:
            query = query.where(UsageEvent.created_at >= since)
        return {status: int(count) for status, count in self.session.execute(query)}

    def recent_events(self, limit: int = 50) -> list[UsageEvent]:
        query = (
            select(UsageEvent).order_by(UsageEvent.created_at.desc(), UsageEvent.id.desc()).limit(limit)
        )
        return list(self.session.execute(query).scalars())

    def ledger_for_event(self, usage_event_id: int) -> list[BudgetLedgerEntry]:
        """The reserve/settle/release trail for one request."""
        query = (
            select(BudgetLedgerEntry)
            .where(BudgetLedgerEntry.usage_event_id == usage_event_id)
            .order_by(BudgetLedgerEntry.id)
        )
        return list(self.session.execute(query).scalars())


def _money(value: float) -> float:
    """Round a summed dollar amount to the registry's cost precision.

    Summing floats accumulates representation error -- $0.01 + $0.05 comes back
    as 0.060000000000000005. Rounding at the query boundary keeps that noise out
    of budget comparisons and dashboard figures. The underlying limitation is the
    ``Float`` column type; see the note in ``app/db/models.py``.
    """
    return round(float(value), COST_PRECISION)


def _percent_saved(baseline: float, actual: float) -> float:
    """Savings as a percentage of the baseline, or 0.0 when nothing ran."""
    if baseline <= 0:
        return 0.0
    return round((baseline - actual) / baseline * 100, 2)


def days_elapsed_this_month(moment: datetime | None = None) -> float:
    """Fractional days since the start of the month, for spend projections."""
    moment = (moment or utcnow()).astimezone(timezone.utc)
    elapsed = moment - start_of_utc_month(moment)
    return max(elapsed / timedelta(days=1), 1 / 24)
