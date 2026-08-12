"""Budget enforcement: boundaries, reconciliation, and concurrency.

The reserve/settle/release cycle is the part of this project most likely to be
wrong in a way that only shows up as a wrong invoice, so the boundaries are
tested exactly and the concurrency case is tested for real rather than reasoned
about.
"""

import asyncio
import os
from datetime import datetime, timezone

import pytest

from app.budget.counters import InMemoryCounters, build_counters
from app.budget.enforcer import BudgetEnforcer, BudgetScope, evaluate
from app.budget.estimator import estimate_cost
from app.budget.policies import BudgetPolicies, Limits, PolicyError
from app.registry import ModelRegistry
from app.schemas import Message

NOON = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def scope(key: str = "day", limit: float = 10.0) -> BudgetScope:
    return BudgetScope(key=key, label=f"{key} budget", limit_usd=limit, ttl_seconds=60)


def policies_file(tmp_path, body: str):
    path = tmp_path / "budgets.yaml"
    path.write_text(body, encoding="utf-8")
    return BudgetPolicies.load(path)


TIGHT_POLICY = """
defaults:
  daily_limit_usd: 10.00
  monthly_limit_usd: 100.00
teams:
  search:
    daily_limit_usd: 1.00
    monthly_limit_usd: 20.00
    features:
      autocomplete:
        daily_limit_usd: 0.40
"""


@pytest.fixture
def enforcer(tmp_path):
    return BudgetEnforcer(
        policies=policies_file(tmp_path, TIGHT_POLICY),
        counters=InMemoryCounters(),
        warn_threshold=0.8,
    )


# --- the pure decision, at its boundaries -----------------------------------


@pytest.mark.parametrize(
    "spent,expected",
    [
        (7.90, "allow"),   # 79% -- below the warning threshold
        (8.00, "warn"),    # exactly 80% -- warning starts here
        (9.99, "warn"),
        (10.00, "warn"),   # exactly at the limit: allowed, you may spend it all
        (10.01, "blocked"),  # past the limit
    ],
)
def test_thresholds_are_exact(spent, expected):
    decision = evaluate(
        totals={"day": spent},
        scopes=[scope(limit=10.0)],
        priority="normal",
        warn_threshold=0.8,
        reserved_usd=0.5,
    )

    assert decision.status == expected


def test_high_priority_overrides_a_broken_limit():
    decision = evaluate(
        totals={"day": 10.50},  # 105% of a $10 limit
        scopes=[scope(limit=10.0)],
        priority="high",
        warn_threshold=0.8,
        reserved_usd=0.5,
    )

    assert decision.status == "override"
    assert decision.allowed
    assert decision.reserved_usd == 0.5  # the reservation is kept
    assert "override" in decision.warnings[0]


@pytest.mark.parametrize("priority", ["low", "normal"])
def test_low_and_normal_priority_are_blocked(priority):
    decision = evaluate(
        totals={"day": 10.50},
        scopes=[scope(limit=10.0)],
        priority=priority,
        warn_threshold=0.8,
        reserved_usd=0.5,
    )

    assert decision.status == "blocked"
    assert not decision.allowed
    assert decision.reserved_usd == 0.0  # nothing is held on a block


def test_a_block_message_names_the_limit_and_the_spend():
    """A blocked caller must be told what happened, not just that it failed."""
    decision = evaluate(
        totals={"day": 12.34},
        scopes=[scope(limit=10.0)],
        priority="normal",
        warn_threshold=0.8,
        reserved_usd=1.0,
    )

    assert "12.34" in decision.reason
    assert "10.00" in decision.reason


def test_the_worst_scope_wins_when_several_are_broken():
    decision = evaluate(
        totals={"day": 11.0, "month": 500.0},
        scopes=[scope("day", limit=10.0), scope("month", limit=100.0)],
        priority="normal",
        warn_threshold=0.8,
        reserved_usd=1.0,
    )

    assert decision.violated_scope == "month"  # 500% beats 110%


def test_percent_used_reports_the_highest_scope():
    decision = evaluate(
        totals={"day": 5.0, "month": 90.0},
        scopes=[scope("day", limit=10.0), scope("month", limit=100.0)],
        priority="normal",
        warn_threshold=0.95,
        reserved_usd=1.0,
    )

    assert decision.percent_used == 90.0


def test_evaluating_with_no_scopes_is_a_programming_error():
    with pytest.raises(ValueError):
        evaluate({}, [], "normal", 0.8, 1.0)


# --- reserve / settle / release ---------------------------------------------


async def test_reserve_then_settle_leaves_the_actual_cost_on_the_counter(enforcer):
    decision = await enforcer.reserve("search", "summaries", "normal", 0.40, now=NOON)
    assert decision.status == "allow"

    await enforcer.settle("search", "summaries", 0.40, 0.25, now=NOON)

    snapshot = await enforcer.snapshot("search", now=NOON)
    assert round(snapshot["daily_spend_usd"], 6) == 0.25


async def test_settle_returns_the_signed_correction(enforcer):
    await enforcer.reserve("search", "summaries", "normal", 0.40, now=NOON)

    handed_back = await enforcer.settle("search", "summaries", 0.40, 0.25, now=NOON)
    assert handed_back == pytest.approx(-0.15)

    await enforcer.reserve("search", "summaries", "normal", 0.10, now=NOON)
    extra = await enforcer.settle("search", "summaries", 0.10, 0.30, now=NOON)
    assert extra == pytest.approx(0.20)


async def test_reserve_then_release_leaves_nothing_behind(enforcer):
    """A provider outage must not consume any of the team's budget."""
    await enforcer.reserve("search", "summaries", "normal", 0.40, now=NOON)

    await enforcer.release("search", "summaries", 0.40, now=NOON)

    snapshot = await enforcer.snapshot("search", now=NOON)
    assert snapshot["daily_spend_usd"] == pytest.approx(0.0)


async def test_a_blocked_request_rolls_back_its_own_reservation(enforcer):
    """Otherwise a stream of rejected requests would exhaust the budget."""
    await enforcer.reserve("search", "summaries", "normal", 0.90, now=NOON)

    blocked = await enforcer.reserve("search", "summaries", "normal", 0.50, now=NOON)
    assert blocked.status == "blocked"

    snapshot = await enforcer.snapshot("search", now=NOON)
    assert round(snapshot["daily_spend_usd"], 6) == 0.90


async def test_an_override_keeps_its_reservation(enforcer):
    await enforcer.reserve("search", "summaries", "normal", 0.95, now=NOON)

    override = await enforcer.reserve("search", "summaries", "high", 0.50, now=NOON)
    assert override.status == "override"

    snapshot = await enforcer.snapshot("search", now=NOON)
    assert round(snapshot["daily_spend_usd"], 6) == 1.45


async def test_negative_estimates_are_rejected(enforcer):
    with pytest.raises(ValueError):
        await enforcer.reserve("search", "summaries", "normal", -1.0, now=NOON)


# --- concurrency -------------------------------------------------------------


async def test_concurrent_reservations_cannot_oversell_the_budget(enforcer):
    """Ten requests race for a $1.00 daily budget at $0.40 each.

    Only two fit. If the check were read-then-write instead of
    increment-then-verify, more than two could pass.
    """
    decisions = await asyncio.gather(
        *[
            enforcer.reserve("search", "summaries", "normal", 0.40, now=NOON)
            for _ in range(10)
        ]
    )

    allowed = [decision for decision in decisions if decision.allowed]
    assert len(allowed) == 2

    snapshot = await enforcer.snapshot("search", now=NOON)
    assert snapshot["daily_spend_usd"] <= snapshot["daily_limit_usd"]


# --- scope resolution --------------------------------------------------------


def test_a_feature_limit_applies_on_top_of_the_team_limit(enforcer):
    scopes = enforcer.scopes_for("search", "autocomplete", now=NOON)

    limits = {s.limit_usd for s in scopes}
    assert 1.00 in limits   # team daily
    assert 0.40 in limits   # feature daily
    assert len(scopes) == 4  # team day+month, feature day+month


def test_a_feature_without_its_own_limit_only_checks_the_team(enforcer):
    scopes = enforcer.scopes_for("search", "summaries", now=NOON)

    assert len(scopes) == 2


async def test_a_feature_cap_blocks_before_the_team_cap(enforcer):
    """autocomplete is capped at $0.40 inside a team allowed $1.00."""
    decision = await enforcer.reserve("search", "autocomplete", "normal", 0.50, now=NOON)

    assert decision.status == "blocked"
    assert "autocomplete" in decision.reason


def test_counter_keys_are_namespaced_by_period(enforcer):
    august = enforcer.scopes_for("search", "summaries", now=NOON)
    september = enforcer.scopes_for(
        "search", "summaries", now=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    )

    assert {s.key for s in august}.isdisjoint({s.key for s in september})
    assert any(s.key.endswith("day:2026-08-12") for s in august)
    assert any(s.key.endswith("month:2026-08") for s in august)


def test_counter_ttls_outlive_their_period(enforcer):
    scopes = enforcer.scopes_for("search", "summaries", now=NOON)
    day = next(s for s in scopes if ":day:" in s.key)

    # Noon leaves 12 hours in the day, plus the grace period.
    assert day.ttl_seconds > 12 * 3600


async def test_unknown_teams_fall_back_to_the_defaults(enforcer):
    snapshot = await enforcer.snapshot("team-that-does-not-exist", now=NOON)

    assert snapshot["daily_limit_usd"] == 10.00


# --- startup seeding ---------------------------------------------------------


async def test_seeding_restores_counters_from_recorded_spend(enforcer):
    """After a restart the counters must reflect the day's real spend.

    Without this, every deploy would hand each team a fresh budget.
    """
    await enforcer.seed("search", daily_spend_usd=0.95, monthly_spend_usd=15.0, now=NOON)

    blocked = await enforcer.reserve("search", "summaries", "normal", 0.30, now=NOON)
    assert blocked.status == "blocked"


# --- counters ----------------------------------------------------------------


async def test_in_memory_counters_add_read_and_overwrite():
    counters = InMemoryCounters()

    totals = await counters.add([("a", 60), ("b", 60)], 1.5)
    assert totals == {"a": 1.5, "b": 1.5}

    await counters.add([("a", 60)], -0.5)
    assert (await counters.get(["a", "b"]))["a"] == 1.0

    await counters.set("a", 9.0, 60)
    assert (await counters.get(["a"]))["a"] == 9.0


async def test_unknown_counter_keys_read_as_zero():
    counters = InMemoryCounters()

    assert await counters.get(["never-written"]) == {"never-written": 0.0}


async def test_expired_counter_keys_are_dropped():
    counters = InMemoryCounters()
    await counters.add([("short-lived", 0)], 5.0)

    assert (await counters.get(["short-lived"]))["short-lived"] == 0.0


def test_backend_selection_follows_configuration():
    assert isinstance(build_counters(None), InMemoryCounters)
    # Constructing the Redis backend does not connect, so this needs no server.
    assert type(build_counters("redis://localhost:6379/0")).__name__ == "RedisCounters"


@pytest.mark.skipif(
    not os.getenv("REDIS_URL"),
    reason="set REDIS_URL to exercise the Redis backend against a real server",
)
async def test_redis_counters_satisfy_the_same_contract():
    """The Redis backend must behave identically to the in-memory one.

    Skipped by default so the suite needs no infrastructure, but run under Docker
    Compose (or with REDIS_URL set) this is what keeps the production backend
    from shipping unverified.
    """
    counters = build_counters(os.environ["REDIS_URL"])
    key = f"test:spend:{os.getpid()}"

    try:
        await counters.set(key, 0.0, 60)

        totals = await counters.add([(key, 60)], 1.5)
        assert totals[key] == pytest.approx(1.5)

        await counters.add([(key, 60)], -0.5)
        assert (await counters.get([key]))[key] == pytest.approx(1.0)

        assert (await counters.get(["test:spend:absent"]))["test:spend:absent"] == 0.0
    finally:
        await counters.set(key, 0.0, 1)
        await counters.close()


# --- policies ----------------------------------------------------------------


def test_features_inherit_limits_they_do_not_override(tmp_path):
    policies = policies_file(tmp_path, TIGHT_POLICY)

    feature = policies.for_feature("search", "autocomplete")
    assert feature is not None
    assert feature.daily_limit_usd == 0.40
    assert feature.monthly_limit_usd == 20.00  # inherited from the team


def test_the_shipped_budgets_file_loads():
    policies = BudgetPolicies.load()

    assert "search" in policies.known_teams()
    assert policies.for_team("search").daily_limit_usd > 0


def test_a_missing_budgets_file_is_reported_clearly(tmp_path):
    with pytest.raises(PolicyError, match="not found"):
        BudgetPolicies.load(tmp_path / "nope.yaml")


def test_defaults_are_required(tmp_path):
    with pytest.raises(PolicyError, match="defaults"):
        policies_file(tmp_path, "teams:\n  search:\n    daily_limit_usd: 1.0\n")


def test_non_positive_limits_are_rejected():
    with pytest.raises(PolicyError):
        Limits(daily_limit_usd=0.0, monthly_limit_usd=10.0)


# --- estimation --------------------------------------------------------------


def test_estimate_prices_a_request_before_it_runs():
    registry = ModelRegistry.load()
    model = registry.get("claude-haiku-4-5")
    messages = [Message(role="user", content="Summarize the Q3 report. " * 50)]

    estimate = estimate_cost(messages, model, max_output_tokens=4096)

    assert estimate.input_tokens > 0
    assert estimate.output_tokens > 0
    assert estimate.cost_usd == model.cost(estimate.input_tokens, estimate.output_tokens)


def test_estimated_output_never_exceeds_the_callers_cap():
    registry = ModelRegistry.load()
    model = registry.get("claude-haiku-4-5")
    messages = [Message(role="user", content="word " * 2000)]

    estimate = estimate_cost(messages, model, max_output_tokens=32)

    assert estimate.output_tokens == 32


def test_a_short_prompt_still_reserves_some_output():
    registry = ModelRegistry.load()
    model = registry.get("claude-haiku-4-5")

    estimate = estimate_cost(
        [Message(role="user", content="hi")], model, max_output_tokens=4096
    )

    assert estimate.output_tokens >= 64


def test_estimate_error_is_signed_and_measurable():
    registry = ModelRegistry.load()
    model = registry.get("claude-haiku-4-5")
    estimate = estimate_cost(
        [Message(role="user", content="hello world")], model, max_output_tokens=4096
    )

    assert estimate.error_against(estimate.cost_usd) == 0.0
    assert estimate.error_against(estimate.cost_usd / 2) > 0  # estimate was high
    assert estimate.error_against(estimate.cost_usd * 2) < 0  # estimate was low
    assert estimate.error_against(0.0) == 0.0  # no division by zero


def test_a_stronger_model_estimates_a_higher_cost():
    registry = ModelRegistry.load()
    messages = [Message(role="user", content="Explain the tax implications. " * 20)]

    cheap = estimate_cost(messages, registry.get("claude-haiku-4-5"), 4096)
    strong = estimate_cost(messages, registry.get("claude-opus-5"), 4096)

    assert strong.cost_usd > cheap.cost_usd
    assert strong.input_tokens == cheap.input_tokens  # same prompt, same tokens
