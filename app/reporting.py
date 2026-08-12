"""Derived metrics: the numbers that are not a single SQL query.

Percentiles, projections, and the savings arithmetic live here. Everything this
module reads comes from ``app/db/repository.py`` -- no SQL is issued directly.

The savings calculation is the most consequential code in the project, so it is
written to be argued with:

    baseline        = every request priced on the strongest model (primary rows only)
    routed          = what the router actually spent
    escalation      = reruns after an unusable answer
    verification    = shadow checks that measured the quality claim

    gross savings   = baseline - routed
    net savings     = baseline - (routed + escalation + verification)

**Net is the honest headline.** Gross savings ignore what the routing cost to
operate, and a system that saves 70% but spends 40% verifying itself has not
saved 70%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.db.repository import Repository, days_elapsed_this_month, start_of_utc_month

#: Days in a month used for projecting month-end spend.
PROJECTION_DAYS = 30


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for an empty sample.

    Nearest-rank rather than interpolated: latency samples are integers in
    milliseconds and an interpolated p95 would invent a value never observed.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return float(ordered[index])


@dataclass
class SavingsSummary:
    """The headline result, with its own operating cost included."""

    requests: int = 0
    """User-facing requests -- the count of primary calls."""
    provider_calls: int = 0
    """Every call that reached a provider, including reruns and verifications."""

    baseline_cost_usd: float = 0.0
    routed_cost_usd: float = 0.0
    escalation_cost_usd: float = 0.0
    verification_cost_usd: float = 0.0

    @property
    def total_spend_usd(self) -> float:
        return round(
            self.routed_cost_usd + self.escalation_cost_usd + self.verification_cost_usd,
            8,
        )

    @property
    def gross_savings_usd(self) -> float:
        return round(self.baseline_cost_usd - self.routed_cost_usd, 8)

    @property
    def net_savings_usd(self) -> float:
        return round(self.baseline_cost_usd - self.total_spend_usd, 8)

    @property
    def gross_savings_percent(self) -> float:
        return _percent(self.gross_savings_usd, self.baseline_cost_usd)

    @property
    def net_savings_percent(self) -> float:
        return _percent(self.net_savings_usd, self.baseline_cost_usd)

    @property
    def overhead_percent_of_routed(self) -> float:
        """Escalation plus verification, as a share of routed spend."""
        return _percent(
            self.escalation_cost_usd + self.verification_cost_usd, self.routed_cost_usd
        )


@dataclass
class VerificationSummary:
    """Measured quality, and what did the measuring."""

    passed: int = 0
    failed: int = 0
    not_run: int = 0
    judges: list[str] = field(default_factory=list)

    @property
    def graded(self) -> int:
        """Checks that produced a verdict. The pass rate's denominator."""
        return self.passed + self.failed

    @property
    def pass_rate_percent(self) -> float:
        return _percent(self.passed, self.graded)

    @property
    def is_mechanical_only(self) -> bool:
        """True when no LLM judge contributed.

        A mechanical-only pass rate is a smoke test, not a quality measurement --
        it detects empty and truncated answers and nothing else. Callers must
        label the number accordingly.
        """
        return bool(self.judges) and all(
            judge.startswith("mechanical") for judge in self.judges
        )

    @property
    def caveat(self) -> str:
        if not self.graded:
            return "No graded checks yet -- this pass rate has no evidence behind it."
        if self.is_mechanical_only:
            return (
                "Mechanical judge only: detects empty and truncated answers, and "
                "cannot assess correctness. Configure a provider API key to grade "
                "with an LLM judge."
            )
        return f"Graded by: {', '.join(self.judges)}"


def savings_summary(repo: Repository, since: datetime | None = None) -> SavingsSummary:
    """Assemble the savings picture, including the cost of producing it."""
    rows = repo.spend_by("kind", since)
    cost_by_kind = {label: cost for label, _count, cost in rows}
    count_by_kind = {label: count for label, count, _cost in rows}

    return SavingsSummary(
        # User-facing requests, which is the primary count. Escalations and
        # shadow checks are extra provider calls against the same requests.
        requests=count_by_kind.get("primary", 0),
        provider_calls=sum(count_by_kind.values()),
        baseline_cost_usd=repo.baseline_total(since),
        routed_cost_usd=cost_by_kind.get("primary", 0.0),
        escalation_cost_usd=cost_by_kind.get("escalation", 0.0),
        verification_cost_usd=cost_by_kind.get("shadow", 0.0),
    )


def verification_summary(
    repo: Repository, since: datetime | None = None
) -> VerificationSummary:
    counts = repo.verification_counts(since)
    return VerificationSummary(
        passed=counts["passed"],
        failed=counts["failed"],
        not_run=counts["not_run"],
        judges=repo.judges_used(since),
    )


def latency_percentiles(
    repo: Repository, since: datetime | None = None
) -> list[dict[str, float | str | int]]:
    """p50 and p95 latency per model, slowest p95 first."""
    grouped: dict[str, list[float]] = {}
    for model, latency in repo.latency_samples(since):
        grouped.setdefault(model, []).append(float(latency))

    rows = [
        {
            "model": model,
            "requests": len(samples),
            "p50_ms": percentile(samples, 0.50),
            "p95_ms": percentile(samples, 0.95),
        }
        for model, samples in grouped.items()
    ]
    return sorted(rows, key=lambda row: row["p95_ms"], reverse=True)


def error_rates(
    repo: Repository, since: datetime | None = None
) -> list[dict[str, float | str | int]]:
    """Attempts and failure rate per provider."""
    totals: dict[str, dict[str, int]] = {}
    for provider, status, count in repo.outcomes_by_provider(since):
        bucket = totals.setdefault(provider, {"attempts": 0, "errors": 0, "blocked": 0})
        bucket["attempts"] += count
        if status == "provider_error":
            bucket["errors"] += count
        elif status == "blocked":
            bucket["blocked"] += count

    return [
        {
            "provider": provider,
            "attempts": counts["attempts"],
            "errors": counts["errors"],
            "blocked": counts["blocked"],
            "error_rate_percent": _percent(counts["errors"], counts["attempts"]),
        }
        for provider, counts in sorted(totals.items())
    ]


def estimate_error(repo: Repository, since: datetime | None = None) -> dict[str, float]:
    """How far the pre-call estimates were from the real costs.

    This is the honesty check on budget enforcement: budgets are held against the
    estimate, so if the estimate is wildly wrong then the enforcement is theatre.
    """
    pairs = repo.estimate_accuracy(since)
    if not pairs:
        return {
            "samples": 0,
            "median_absolute_percent": 0.0,
            "median_signed_percent": 0.0,
            "mean_absolute_percent": 0.0,
        }

    signed = [(estimated - actual) / actual * 100 for estimated, actual in pairs]
    absolute = [abs(value) for value in signed]

    return {
        "samples": len(pairs),
        # The median is the headline. Percentage error divides by the actual cost,
        # so a single request that returned almost no output -- an empty answer,
        # say -- produces an error in the thousands of percent and drags the mean
        # with it. Observed live: a median of -8.5% alongside a mean of 1495%,
        # from one degenerate response out of four. The mean is kept for
        # completeness, not for quoting.
        "median_absolute_percent": round(percentile(absolute, 0.50), 2),
        "median_signed_percent": round(percentile(signed, 0.50), 2),
        "mean_absolute_percent": round(sum(absolute) / len(absolute), 2),
    }


def month_end_projection(repo: Repository, team_id: str) -> dict[str, float]:
    """Extrapolate this month's spend to month end at the current daily rate."""
    spent = repo.spend_this_month(team_id)
    elapsed = days_elapsed_this_month()
    daily_rate = spent / elapsed

    return {
        "spent_so_far_usd": spent,
        "days_elapsed": round(elapsed, 2),
        "daily_rate_usd": round(daily_rate, 8),
        "projected_month_end_usd": round(daily_rate * PROJECTION_DAYS, 8),
    }


def escalation_rate(repo: Repository, since: datetime | None = None) -> float:
    """Share of routed requests that were rerun on a stronger model."""
    counts = {label: count for label, count, _ in repo.spend_by("kind", since)}
    return _percent(counts.get("escalation", 0), counts.get("primary", 0))


def month_to_date(repo: Repository) -> SavingsSummary:
    return savings_summary(repo, since=start_of_utc_month())


def _percent(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round(part / whole * 100, 2)
