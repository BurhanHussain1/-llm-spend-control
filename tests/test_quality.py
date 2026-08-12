"""Sampling, judging, escalation, and shadow verification.

These are the tests that decide whether the savings number is defensible, so the
metric definitions get as much attention as the behaviour: a skipped check must
not count as a failure, and verification cost must not hide inside routed spend.
"""

import pytest

from app.budget.counters import InMemoryCounters
from app.budget.enforcer import BudgetEnforcer
from app.budget.policies import BudgetPolicies
from app.db.engine import create_tables, get_session_factory, reset_engine
from app.db.models import RoutingMiss
from app.db.repository import Repository
from app.providers.base import Completion, ProviderError
from app.providers.mock import MockProvider
from app.quality.judge import LLMJudge, MechanicalJudge, _parse_grade
from app.quality.sampler import should_escalate, should_shadow_check
from app.quality.shadow import ShadowJob, ShadowVerifier
from app.registry import ModelRegistry
from app.schemas import ChatRequest, Message
from app.settings import get_settings

GENEROUS_POLICY = """
defaults:
  daily_limit_usd: 100.00
  monthly_limit_usd: 1000.00
teams:
  broke:
    daily_limit_usd: 0.0000001
    monthly_limit_usd: 0.0000002
"""


@pytest.fixture
def registry():
    return ModelRegistry.load()


@pytest.fixture
def enforcer(tmp_path):
    path = tmp_path / "budgets.yaml"
    path.write_text(GENEROUS_POLICY, encoding="utf-8")
    return BudgetEnforcer(BudgetPolicies.load(path), InMemoryCounters(), 0.8)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'q.db').as_posix()}")
    get_settings.cache_clear()
    reset_engine()
    create_tables()
    yield
    reset_engine()
    get_settings.cache_clear()


def request_for(content: str = "Extract the total.", **overrides) -> ChatRequest:
    body = {
        "messages": [Message(role="user", content=content)],
        "team_id": "search",
        "feature": "summaries",
    }
    body.update(overrides)
    return ChatRequest(**body)


def completion(text: str = "an answer of reasonable length", tokens: int = 40, **extra):
    return Completion(
        text=text,
        input_tokens=20,
        output_tokens=tokens,
        latency_ms=100,
        provider_metadata=extra,
    )


# --- sampling ----------------------------------------------------------------


def test_sampling_is_deterministic(registry):
    """The same workload must verify the same requests on every run."""
    model = registry.get("mock-cheap")
    prompt = "Summarize the quarterly report."

    first = should_shadow_check(prompt, model, strongest_tier=3, rate=0.5)
    again = should_shadow_check(prompt, model, strongest_tier=3, rate=0.5)

    assert first == again


def test_sampling_rate_is_roughly_honoured(registry):
    model = registry.get("mock-cheap")
    prompts = [f"Extract field {index} from this record." for index in range(2000)]

    sampled = sum(
        should_shadow_check(p, model, strongest_tier=3, rate=0.1) for p in prompts
    )

    assert 0.07 < sampled / len(prompts) < 0.13


def test_the_strongest_model_is_never_shadow_checked(registry):
    """Verifying the reference model against itself costs money and proves nothing."""
    strong = registry.get("mock-strong")

    assert not should_shadow_check("anything", strong, strongest_tier=3, rate=1.0)


def test_a_zero_rate_disables_sampling(registry):
    model = registry.get("mock-cheap")

    assert not should_shadow_check("anything", model, strongest_tier=3, rate=0.0)


# --- escalation decisions ----------------------------------------------------


def test_an_empty_answer_triggers_escalation(registry):
    decision = should_escalate(
        request_for(),
        completion(text="", tokens=0),
        registry.get("mock-cheap"),
        strongest_tier=3,
        escalate_high_priority=False,
    )

    assert decision.needed
    assert "empty" in decision.reason


def test_a_stub_answer_triggers_escalation(registry):
    decision = should_escalate(
        request_for(),
        completion(text="ok", tokens=1),
        registry.get("mock-cheap"),
        strongest_tier=3,
        escalate_high_priority=False,
    )

    assert decision.needed
    assert "too short" in decision.reason


def test_a_refusal_triggers_escalation(registry):
    decision = should_escalate(
        request_for(),
        completion(text="", tokens=0, refused=True),
        registry.get("mock-cheap"),
        strongest_tier=3,
        escalate_high_priority=False,
    )

    assert decision.needed


def test_a_good_answer_is_left_alone(registry):
    decision = should_escalate(
        request_for(),
        completion(),
        registry.get("mock-cheap"),
        strongest_tier=3,
        escalate_high_priority=True,
    )

    assert not decision.needed


def test_high_priority_escalates_only_from_tier_one(registry):
    """Rerunning everything urgent would double its cost for little gain."""
    from_tier_one = should_escalate(
        request_for(priority="high"),
        completion(),
        registry.get("mock-cheap"),
        strongest_tier=3,
        escalate_high_priority=True,
    )
    from_tier_two = should_escalate(
        request_for(priority="high"),
        completion(),
        registry.get("mock-mid"),
        strongest_tier=3,
        escalate_high_priority=True,
    )

    assert from_tier_one.needed
    assert not from_tier_two.needed


def test_high_priority_escalation_can_be_switched_off(registry):
    decision = should_escalate(
        request_for(priority="high"),
        completion(),
        registry.get("mock-cheap"),
        strongest_tier=3,
        escalate_high_priority=False,
    )

    assert not decision.needed


def test_the_strongest_model_is_never_escalated(registry):
    decision = should_escalate(
        request_for(priority="high"),
        completion(text="", tokens=0),
        registry.get("mock-strong"),
        strongest_tier=3,
        escalate_high_priority=True,
    )

    assert not decision.needed


# --- judges ------------------------------------------------------------------


async def test_the_mechanical_judge_fails_an_empty_answer():
    verdict = await MechanicalJudge().grade("prompt", "", "a full reference answer")

    assert not verdict.passed
    assert verdict.judge == "mechanical"


async def test_the_mechanical_judge_fails_a_truncated_answer():
    verdict = await MechanicalJudge().grade("prompt", "The", "A" * 400)

    assert not verdict.passed
    assert "stopped early" in verdict.reason


async def test_the_mechanical_judge_passes_a_comparable_answer():
    verdict = await MechanicalJudge().grade("prompt", "B" * 350, "A" * 400)

    assert verdict.passed


async def test_the_mechanical_judge_admits_what_it_cannot_check():
    """Its verdicts must not read as quality measurements."""
    verdict = await MechanicalJudge().grade("prompt", "B" * 350, "A" * 400)

    assert "cannot assess correctness" in verdict.reason


async def test_an_empty_reference_is_not_scored_as_a_pass():
    verdict = await MechanicalJudge().grade("prompt", "an answer", "")

    assert not verdict.passed
    assert "reference" in verdict.reason


@pytest.mark.parametrize(
    "text,expected",
    [
        ("SCORE: 0.85\nREASON: missed a caveat", 0.85),
        ("score: 1.0\nreason: identical", 1.0),
        ("SCORE: 1.7\nREASON: over range", 1.0),  # clamped
        ("SCORE: -3\nREASON: under range", 0.0),  # clamped
    ],
)
def test_judge_grades_are_parsed_and_clamped(text, expected):
    score, _ = _parse_grade(text)

    assert score == expected


def test_an_unparseable_grade_is_detected():
    score, _ = _parse_grade("I think it was pretty good actually")

    assert score is None


async def test_an_unparseable_llm_grade_is_not_counted_as_a_pass(registry):
    """Silently passing a broken judge response would inflate the pass rate."""

    class ChattyProvider:
        name = "mock"

        async def grade(self, *_args, **_kwargs): ...

        async def complete(self, model, messages, max_output_tokens):
            return completion(text="Looks fine to me!")

    judge = LLMJudge(registry.get("mock-strong"), ChattyProvider())

    verdict = await judge.grade("prompt", "candidate", "reference")

    assert not verdict.passed
    assert "unparseable" in verdict.judge


async def test_a_judge_outage_is_reported_not_swallowed(registry):
    class BrokenProvider:
        name = "mock"

        async def complete(self, model, messages, max_output_tokens):
            raise ProviderError("mock", "judge is down")

    judge = LLMJudge(registry.get("mock-strong"), BrokenProvider())

    verdict = await judge.grade("prompt", "candidate", "reference")

    assert not verdict.passed
    assert verdict.judge.endswith(":error")


async def test_the_llm_judge_applies_its_threshold(registry):
    class ScoringProvider:
        def __init__(self, score):
            self.score = score

        async def complete(self, model, messages, max_output_tokens):
            return completion(text=f"SCORE: {self.score}\nREASON: because")

    strong = registry.get("mock-strong")

    assert (await LLMJudge(strong, ScoringProvider(0.9), 0.7).grade("p", "c", "r")).passed
    assert not (
        await LLMJudge(strong, ScoringProvider(0.5), 0.7).grade("p", "c", "r")
    ).passed


# --- shadow verification end to end -----------------------------------------


def job(**overrides) -> ShadowJob:
    body = {
        "team_id": "search",
        "feature": "summaries",
        "priority": "normal",
        "messages": [Message(role="user", content="Summarize the Q3 report.")],
        "candidate_text": "A short cheap answer.",
        "candidate_model": "mock-cheap",
        "usage_event_id": 1,
        "max_output_tokens": 512,
    }
    body.update(overrides)
    return ShadowJob(**body)


def checks() -> list[RoutingMiss]:
    session = get_session_factory()()
    try:
        return session.query(RoutingMiss).order_by(RoutingMiss.id).all()
    finally:
        session.close()


def shadow_rows():
    session = get_session_factory()()
    try:
        return [e for e in Repository(session).recent_events() if e.kind == "shadow"]
    finally:
        session.close()


def verifier_for(registry, enforcer, judge=None) -> ShadowVerifier:
    return ShadowVerifier(
        registry=registry,
        reference_model=registry.get("mock-strong"),
        provider=MockProvider(registry),
        enforcer=enforcer,
        judge=judge or MechanicalJudge(),
    )


async def test_verification_records_a_verdict(db, registry, enforcer):
    verdict = await verifier_for(registry, enforcer).run(job())

    assert verdict is not None
    stored = checks()
    assert len(stored) == 1
    assert stored[0].chosen_model == "mock-cheap"
    assert stored[0].better_model == "mock-strong"
    assert stored[0].passed is verdict.passed


async def test_the_verdict_names_the_judge_that_produced_it(db, registry, enforcer):
    """A pass rate is meaningless without knowing what did the judging."""
    await verifier_for(registry, enforcer).run(job())

    assert "[mechanical]" in checks()[0].reason


async def test_verification_cost_is_logged_separately_from_routed_spend(
    db, registry, enforcer
):
    """Verification is real money and must not masquerade as routing spend."""
    await verifier_for(registry, enforcer).run(job())

    rows = shadow_rows()
    assert len(rows) == 1
    assert rows[0].cost_usd > 0
    # No counterfactual: this IS the baseline model, so claiming savings on it
    # would double count.
    assert rows[0].baseline_cost_usd == 0.0


async def test_verification_charges_the_teams_budget(db, registry, enforcer):
    before = (await enforcer.snapshot("search"))["daily_spend_usd"]

    await verifier_for(registry, enforcer).run(job())

    after = (await enforcer.snapshot("search"))["daily_spend_usd"]
    assert after > before


async def test_a_cheap_answer_that_looks_truncated_is_recorded_as_a_miss(
    db, registry, enforcer
):
    verdict = await verifier_for(registry, enforcer).run(
        job(candidate_text="Th")  # far shorter than the reference
    )

    assert verdict is not None and not verdict.passed
    assert checks()[0].passed is False


async def test_verification_refused_by_the_budget_is_not_counted_as_a_failure(
    db, registry, enforcer
):
    """The critical metric rule: a check that never ran is not a bad answer."""
    verdict = await verifier_for(registry, enforcer).run(job(team_id="broke"))

    assert verdict is None
    stored = checks()
    assert len(stored) == 1
    assert stored[0].passed is None  # not False
    assert "[not-run]" in stored[0].reason


async def test_a_reference_model_outage_is_not_counted_as_a_failure(
    db, registry, enforcer
):
    class BrokenProvider:
        name = "mock"

        async def complete(self, model, messages, max_output_tokens):
            raise ProviderError("mock", "reference model down")

    verifier = ShadowVerifier(
        registry=registry,
        reference_model=registry.get("mock-strong"),
        provider=BrokenProvider(),
        enforcer=enforcer,
        judge=MechanicalJudge(),
    )

    verdict = await verifier.run(job())

    assert verdict is None
    assert checks()[0].passed is None


async def test_a_failed_reference_call_refunds_its_reservation(db, registry, enforcer):
    class BrokenProvider:
        name = "mock"

        async def complete(self, model, messages, max_output_tokens):
            raise ProviderError("mock", "reference model down")

    verifier = ShadowVerifier(
        registry=registry,
        reference_model=registry.get("mock-strong"),
        provider=BrokenProvider(),
        enforcer=enforcer,
        judge=MechanicalJudge(),
    )

    await verifier.run(job())

    assert (await enforcer.snapshot("search"))["daily_spend_usd"] == pytest.approx(0.0)

