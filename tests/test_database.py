"""The usage log records what happened and totals it correctly.

The spend queries here are what budget enforcement reads, and the savings totals
are what the case study quotes, so both are tested against seeded rows with
hand-checked arithmetic.
"""

from datetime import timedelta

from app.db.models import utcnow
from app.db.repository import (
    days_elapsed_this_month,
    start_of_utc_day,
    start_of_utc_month,
)


def log(repo, **overrides):
    """Log a billable usage event, overriding only what a test cares about."""
    fields = {
        "team_id": "search",
        "feature": "autocomplete",
        "priority": "normal",
        "model": "mock-cheap",
        "provider": "mock",
        "tier": 1,
        "routing_reason": "classifier: tier 1",
        "status": "ok",
        "input_tokens": 1000,
        "output_tokens": 500,
        "latency_ms": 900,
        "cost_usd": 0.0035,
        "estimated_cost_usd": 0.0040,
        "baseline_cost_usd": 0.0175,
    }
    fields.update(overrides)
    return repo.log_usage(**fields)


# --- round trip --------------------------------------------------------------


def test_a_usage_event_can_be_written_and_read_back(repo, db_session):
    event = log(repo, prompt="Summarize the Q3 report.")
    db_session.commit()

    stored = repo.recent_events()[0]
    assert stored.id == event.id
    assert stored.team_id == "search"
    assert stored.model == "mock-cheap"
    assert stored.cost_usd == 0.0035
    assert stored.baseline_cost_usd == 0.0175
    assert stored.prompt_preview == "Summarize the Q3 report."


def test_long_prompts_are_truncated_not_stored_whole(repo, db_session):
    log(repo, prompt="x" * 5000)
    db_session.commit()

    assert len(repo.recent_events()[0].prompt_preview) == 500


# --- spend queries -----------------------------------------------------------


def test_spend_today_sums_only_this_team(repo, db_session):
    log(repo, team_id="search", cost_usd=0.01)
    log(repo, team_id="search", cost_usd=0.02)
    log(repo, team_id="billing", cost_usd=0.99)
    db_session.commit()

    assert repo.spend_today("search") == 0.03
    assert repo.spend_today("billing") == 0.99


def test_spend_can_be_narrowed_to_one_feature(repo, db_session):
    log(repo, feature="autocomplete", cost_usd=0.01)
    log(repo, feature="summaries", cost_usd=0.05)
    db_session.commit()

    assert repo.spend_today("search", feature="summaries") == 0.05
    assert repo.spend_today("search") == 0.06


def test_blocked_and_errored_attempts_are_recorded_but_not_billed(repo, db_session):
    """The audit trail needs failed attempts; spend totals must not include them."""
    log(repo, cost_usd=0.01)
    log(repo, status="blocked", cost_usd=0.0)
    log(repo, status="provider_error", cost_usd=0.0, error="upstream 503")
    db_session.commit()

    assert repo.spend_today("search") == 0.01
    assert repo.count_by_status() == {"ok": 1, "blocked": 1, "provider_error": 1}
    assert len(repo.recent_events()) == 3


def test_yesterdays_spend_does_not_count_against_todays_budget(repo, db_session):
    log(repo, cost_usd=5.00, created_at=start_of_utc_day() - timedelta(minutes=1))
    log(repo, cost_usd=0.25)
    db_session.commit()

    assert repo.spend_today("search") == 0.25


def test_monthly_spend_includes_earlier_days_in_the_month(repo, db_session):
    """A row from earlier today's month counts monthly but not daily."""
    earlier = start_of_utc_month() + timedelta(hours=1)
    log(repo, cost_usd=1.50, created_at=earlier)
    log(repo, cost_usd=0.25)
    db_session.commit()

    monthly = repo.spend_this_month("search")
    assert round(monthly, 4) == 1.75
    assert monthly >= repo.spend_today("search")


def test_last_months_spend_is_excluded_from_this_month(repo, db_session):
    log(repo, cost_usd=99.0, created_at=start_of_utc_month() - timedelta(minutes=1))
    log(repo, cost_usd=0.25)
    db_session.commit()

    assert repo.spend_this_month("search") == 0.25


# --- savings totals ----------------------------------------------------------


def test_totals_compute_savings_against_the_baseline(repo, db_session):
    # Two requests: $0.10 actual against $1.00 baseline = 90% saved.
    log(repo, cost_usd=0.04, baseline_cost_usd=0.40, estimated_cost_usd=0.05)
    log(repo, cost_usd=0.06, baseline_cost_usd=0.60, estimated_cost_usd=0.06)
    db_session.commit()

    totals = repo.totals()

    assert totals["requests"] == 2
    assert round(totals["actual_cost_usd"], 4) == 0.10
    assert round(totals["baseline_cost_usd"], 4) == 1.00
    assert round(totals["savings_usd"], 4) == 0.90
    assert totals["savings_percent"] == 90.0


def test_totals_on_an_empty_database_report_zero_not_a_crash(repo):
    totals = repo.totals()

    assert totals["requests"] == 0
    assert totals["savings_percent"] == 0.0


def test_totals_keep_the_estimate_so_estimate_error_is_measurable(repo, db_session):
    log(repo, cost_usd=0.10, estimated_cost_usd=0.15, baseline_cost_usd=0.50)
    db_session.commit()

    totals = repo.totals()
    assert round(totals["estimated_cost_usd"], 4) == 0.15
    assert round(totals["actual_cost_usd"], 4) == 0.10


# --- ledger ------------------------------------------------------------------


def test_the_ledger_records_a_full_reserve_then_settle_trail(repo, db_session):
    event = log(repo, cost_usd=0.0035, estimated_cost_usd=0.0040)
    repo.record_ledger(
        team_id="search",
        feature="autocomplete",
        action="reserve",
        amount_usd=0.0040,
        usage_event_id=event.id,
        note="pre-call estimate",
    )
    repo.record_ledger(
        team_id="search",
        feature="autocomplete",
        action="settle",
        amount_usd=-0.0005,
        usage_event_id=event.id,
        note="estimate was high by $0.0005",
    )
    db_session.commit()

    trail = repo.ledger_for_event(event.id)
    assert [entry.action for entry in trail] == ["reserve", "settle"]
    # Net movement equals the real cost, which is the point of settling.
    assert round(sum(entry.amount_usd for entry in trail), 6) == 0.0035


def test_a_released_reservation_nets_to_zero(repo, db_session):
    """A failed provider call must leave the team's budget untouched."""
    event = log(repo, status="provider_error", cost_usd=0.0, error="timeout")
    repo.record_ledger(
        team_id="search",
        feature="autocomplete",
        action="reserve",
        amount_usd=0.004,
        usage_event_id=event.id,
    )
    repo.record_ledger(
        team_id="search",
        feature="autocomplete",
        action="release",
        amount_usd=-0.004,
        usage_event_id=event.id,
        note="provider error",
    )
    db_session.commit()

    trail = repo.ledger_for_event(event.id)
    assert sum(entry.amount_usd for entry in trail) == 0.0


# --- quality checks ----------------------------------------------------------


def test_quality_checks_store_passes_as_well_as_misses(repo, db_session):
    """The verifier pass rate needs a denominator, so passes are stored too."""
    repo.record_quality_check(
        prompt="Extract the invoice date.",
        chosen_model="mock-cheap",
        better_model="mock-strong",
        passed=True,
        score=0.95,
    )
    repo.record_quality_check(
        prompt="Explain the tax implications.",
        chosen_model="mock-cheap",
        better_model="mock-strong",
        passed=False,
        score=0.31,
        reason="missed the withholding rule",
    )
    db_session.commit()

    from app.db.models import RoutingMiss

    checks = db_session.query(RoutingMiss).order_by(RoutingMiss.id).all()
    assert [check.passed for check in checks] == [True, False]
    assert checks[1].reason == "missed the withholding rule"


# --- period boundaries -------------------------------------------------------


def test_start_of_day_and_month_are_midnight_utc():
    day = start_of_utc_day()
    month = start_of_utc_month()

    assert (day.hour, day.minute, day.second, day.microsecond) == (0, 0, 0, 0)
    assert month.day == 1
    assert month <= day


def test_days_elapsed_this_month_is_never_zero():
    """Guards the month-end projection from dividing by zero on the 1st."""
    assert days_elapsed_this_month(start_of_utc_month()) > 0
    assert days_elapsed_this_month(utcnow()) >= 1 / 24
