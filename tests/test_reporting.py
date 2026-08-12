"""Reporting arithmetic.

The savings figure is the project's headline claim, so its definition is tested
directly against hand-checked numbers rather than only through the dashboard.
"""

import pytest

from app import reporting
from app.reporting import SavingsSummary, VerificationSummary, percentile


def log(repo, **overrides):
    fields = {
        "team_id": "search",
        "feature": "summaries",
        "priority": "normal",
        "model": "mock-cheap",
        "provider": "mock",
        "tier": 1,
        "routing_reason": "classifier",
        "status": "ok",
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 900,
        "cost_usd": 0.10,
        "estimated_cost_usd": 0.12,
        "baseline_cost_usd": 1.00,
        "kind": "primary",
    }
    fields.update(overrides)
    return repo.log_usage(**fields)


# --- percentiles -------------------------------------------------------------


def test_percentile_of_an_empty_sample_is_zero():
    assert percentile([], 0.95) == 0.0


def test_percentiles_return_observed_values():
    """Nearest-rank, so a p95 is always a latency that actually happened."""
    values = [float(n) for n in range(1, 101)]

    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 1.0) == 100.0


# --- savings arithmetic ------------------------------------------------------


def test_net_savings_subtract_escalation_and_verification():
    """Hand-checked: baseline 100, routed 30, escalation 5, verification 15."""
    summary = SavingsSummary(
        requests=10,
        provider_calls=13,
        baseline_cost_usd=100.0,
        routed_cost_usd=30.0,
        escalation_cost_usd=5.0,
        verification_cost_usd=15.0,
    )

    assert summary.total_spend_usd == 50.0
    assert summary.gross_savings_usd == 70.0
    assert summary.gross_savings_percent == 70.0
    assert summary.net_savings_usd == 50.0
    assert summary.net_savings_percent == 50.0


def test_overhead_is_expressed_against_routed_spend():
    summary = SavingsSummary(
        baseline_cost_usd=100.0,
        routed_cost_usd=40.0,
        escalation_cost_usd=4.0,
        verification_cost_usd=16.0,
    )

    assert summary.overhead_percent_of_routed == 50.0


def test_savings_can_be_negative_when_overhead_exceeds_the_gain():
    """The report must be able to say routing lost money."""
    summary = SavingsSummary(
        baseline_cost_usd=10.0,
        routed_cost_usd=4.0,
        escalation_cost_usd=3.0,
        verification_cost_usd=8.0,
    )

    assert summary.gross_savings_percent == 60.0
    assert summary.net_savings_usd < 0
    assert summary.net_savings_percent < 0


def test_an_empty_summary_reports_zero_not_a_crash():
    assert SavingsSummary().net_savings_percent == 0.0


def test_savings_are_summed_from_the_usage_log(repo, db_session):
    # Two routed requests, one escalation, one shadow check.
    log(repo, cost_usd=0.10, baseline_cost_usd=1.00)
    log(repo, cost_usd=0.20, baseline_cost_usd=2.00)
    log(repo, kind="escalation", model="mock-strong", cost_usd=0.50, baseline_cost_usd=0.0)
    log(repo, kind="shadow", model="mock-strong", cost_usd=0.40, baseline_cost_usd=0.0)
    db_session.commit()

    summary = reporting.savings_summary(repo)

    assert summary.requests == 2  # user-facing requests, not provider calls
    assert summary.provider_calls == 4
    assert summary.baseline_cost_usd == 3.00
    assert summary.routed_cost_usd == 0.30
    assert summary.escalation_cost_usd == 0.50
    assert summary.verification_cost_usd == 0.40
    assert summary.total_spend_usd == 1.20
    assert summary.net_savings_usd == 1.80


def test_only_primary_rows_contribute_a_baseline(repo, db_session):
    """Otherwise an escalated request would count its counterfactual twice."""
    log(repo, cost_usd=0.10, baseline_cost_usd=1.00)
    log(repo, kind="escalation", cost_usd=1.00, baseline_cost_usd=1.00)  # wrong on purpose
    db_session.commit()

    assert repo.baseline_total() == 1.00


def test_blocked_requests_do_not_affect_savings(repo, db_session):
    log(repo, cost_usd=0.10, baseline_cost_usd=1.00)
    log(repo, status="blocked", cost_usd=0.0, baseline_cost_usd=1.00)
    db_session.commit()

    summary = reporting.savings_summary(repo)

    assert summary.requests == 1
    assert summary.baseline_cost_usd == 1.00


# --- verification reporting --------------------------------------------------


def test_skipped_checks_are_excluded_from_the_pass_rate():
    """The metric rule that matters: not-run is not a failure."""
    summary = VerificationSummary(passed=8, failed=2, not_run=90)

    assert summary.graded == 10
    assert summary.pass_rate_percent == 80.0


def test_a_mechanical_only_pass_rate_carries_a_warning():
    summary = VerificationSummary(passed=9, failed=1, judges=["mechanical"])

    assert summary.is_mechanical_only
    assert "cannot assess correctness" in summary.caveat


def test_an_llm_judged_pass_rate_names_the_judge():
    summary = VerificationSummary(passed=9, failed=1, judges=["llm:claude-opus-5"])

    assert not summary.is_mechanical_only
    assert "claude-opus-5" in summary.caveat


def test_no_graded_checks_says_so_plainly():
    summary = VerificationSummary(passed=0, failed=0, not_run=5)

    assert summary.pass_rate_percent == 0.0
    assert "no evidence" in summary.caveat


def test_verification_counts_come_back_split_three_ways(repo, db_session):
    for passed in (True, True, False, None):
        repo.record_quality_check(
            prompt="p",
            chosen_model="mock-cheap",
            better_model="mock-strong",
            passed=passed,
            reason="[mechanical] whatever",
        )
    db_session.commit()

    counts = repo.verification_counts()

    assert counts == {"passed": 2, "failed": 1, "not_run": 1}


def test_the_judge_is_read_back_from_stored_verdicts(repo, db_session):
    repo.record_quality_check(
        prompt="p",
        chosen_model="mock-cheap",
        better_model="mock-strong",
        passed=True,
        reason="[llm:mock-strong] close enough",
    )
    db_session.commit()

    assert repo.judges_used() == ["llm:mock-strong"]


# --- operational metrics -----------------------------------------------------


def test_latency_percentiles_are_grouped_by_model(repo, db_session):
    for latency in (100, 200, 300, 400, 900):
        log(repo, model="mock-cheap", latency_ms=latency)
    log(repo, model="mock-strong", latency_ms=5000)
    db_session.commit()

    rows = reporting.latency_percentiles(repo)

    assert rows[0]["model"] == "mock-strong"  # sorted by slowest p95
    cheap = next(row for row in rows if row["model"] == "mock-cheap")
    assert cheap["requests"] == 5
    assert cheap["p95_ms"] == 900


def test_error_rates_separate_provider_failures_from_budget_blocks(repo, db_session):
    """A blocked request never reached a provider and must not count against it."""
    log(repo)
    log(repo, status="provider_error", cost_usd=0.0)
    log(repo, status="blocked", cost_usd=0.0)
    db_session.commit()

    rates = reporting.error_rates(repo)
    mock = next(row for row in rates if row["provider"] == "mock")

    assert mock["attempts"] == 3
    assert mock["errors"] == 1
    assert mock["blocked"] == 1
    assert mock["error_rate_percent"] == pytest.approx(33.33, abs=0.01)


def test_estimate_error_is_signed_and_absolute(repo, db_session):
    log(repo, estimated_cost_usd=0.12, cost_usd=0.10)  # +20%
    log(repo, estimated_cost_usd=0.08, cost_usd=0.10)  # -20%
    db_session.commit()

    error = reporting.estimate_error(repo)

    assert error["samples"] == 2
    assert error["mean_absolute_percent"] == pytest.approx(20.0, abs=0.01)


def test_one_degenerate_response_cannot_wreck_the_estimate_error_headline(
    repo, db_session
):
    """Percentage error divides by actual cost, so a near-zero actual explodes.

    Observed live: a single empty answer produced a mean absolute error of 1495%
    while the typical request was within 10%. The median has to be the headline.
    """
    for _ in range(9):
        log(repo, estimated_cost_usd=0.11, cost_usd=0.10)  # +10%
    log(repo, estimated_cost_usd=0.10, cost_usd=0.0001)  # empty answer: +99,900%
    db_session.commit()

    error = reporting.estimate_error(repo)

    assert error["median_absolute_percent"] == pytest.approx(10.0, abs=0.01)
    assert error["mean_absolute_percent"] > 1000  # the mean is useless here


def test_escalation_rate_is_a_share_of_routed_requests(repo, db_session):
    for _ in range(4):
        log(repo)
    log(repo, kind="escalation", cost_usd=0.5, baseline_cost_usd=0.0)
    db_session.commit()

    assert reporting.escalation_rate(repo) == 25.0


def test_spend_can_be_grouped_several_ways(repo, db_session):
    log(repo, team_id="search", feature="autocomplete", cost_usd=0.10)
    log(repo, team_id="billing", feature="disputes", cost_usd=0.90)
    db_session.commit()

    assert repo.spend_by("team")[0] == ("billing", 1, 0.90)
    assert dict((row[0], row[2]) for row in repo.spend_by("feature")) == {
        "autocomplete": 0.10,
        "disputes": 0.90,
    }


def test_grouping_by_an_unknown_column_is_rejected(repo):
    with pytest.raises(ValueError, match="cannot group by"):
        repo.spend_by("favourite_colour")


def test_month_end_projection_extrapolates_the_daily_rate(repo, db_session):
    log(repo, cost_usd=1.00)
    db_session.commit()

    projection = reporting.month_end_projection(repo, "search")

    assert projection["spent_so_far_usd"] == 1.00
    assert projection["days_elapsed"] > 0
    assert projection["projected_month_end_usd"] >= 1.00
