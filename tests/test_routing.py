"""Classifier behaviour and model selection.

The classifier's accuracy is measured by ``eval/classifier_eval.py``, not here.
These tests pin the behaviour that must not regress silently: risk tags win,
overrides win, and an unrecognised prompt lands in the middle tier rather than
the cheapest one.
"""

import pytest

from app.registry import ModelRegistry, RegistryError
from app.routing.classifier import classify
from app.routing.router import Router, RoutingError, RoutingPolicy
from app.schemas import ChatRequest, Message


def request(
    content: str = "Summarize this thread.",
    *,
    feature: str = "summaries",
    priority: str = "normal",
    risk_tags: list[str] | None = None,
    preferred_model: str | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content=content)],
        team_id="search",
        feature=feature,
        priority=priority,
        risk_tags=risk_tags or [],
        preferred_model=preferred_model,
    )


@pytest.fixture
def policy():
    return RoutingPolicy.load()


@pytest.fixture
def mock_only_router(policy):
    """A router with no provider credentials, as a fresh clone would run."""
    return Router(ModelRegistry.load(), policy, available_providers={"mock"})


@pytest.fixture
def full_router(policy):
    return Router(
        ModelRegistry.load(),
        policy,
        available_providers={"mock", "openai", "anthropic", "ollama"},
    )


# --- classifier behaviour that must not regress ------------------------------


def test_an_unrecognised_prompt_lands_in_the_middle_tier():
    """The most important classifier property.

    An unknown prompt must not default to the cheapest model -- under-routing is
    the expensive kind of mistake. Holdout evaluation caught this defaulting to
    tier 1 and it cost two-thirds of the tier-3 recall.
    """
    result = classify("Kindly attend to the matter enclosed herein.")

    assert result.tier == 2


def test_extraction_prompts_reach_tier_one():
    assert classify("Extract the invoice number from this email.").tier == 1


def test_reasoning_prompts_reach_tier_three():
    assert classify("Diagnose why this deployment rolled back.").tier == 3


def test_risk_tags_outrank_a_trivial_looking_prompt():
    """A textually simple question can still be legally consequential."""
    plain = classify("Extract the refund amount.")
    tagged = classify("Extract the refund amount.", risk_tags=["legal"])

    assert plain.tier == 1
    assert tagged.tier == 3


def test_the_classification_explains_itself():
    result = classify("Analyze this funnel and explain the drop-off.")

    assert result.reasons
    assert "tier 3" in result.summary
    assert "reasoning verb" in result.summary


def test_a_long_prompt_is_pushed_upward():
    short = classify("Extract the total.")
    padded = classify("Extract the total. " + ("context line here. " * 400))

    assert padded.score > short.score


# --- tier resolution ---------------------------------------------------------


def test_the_classifier_decides_when_nothing_overrides_it(full_router):
    decision = full_router.route(request("Extract the invoice number."))

    assert decision.tier == 1
    assert decision.classification is not None
    assert "classifier" in decision.reason


def test_a_feature_override_beats_the_classifier(full_router):
    """payment_dispute is pinned to tier 3 however trivial the text looks."""
    decision = full_router.route(
        request("Extract the disputed amount.", feature="payment_dispute")
    )

    assert decision.tier == 3
    assert "feature override" in decision.reason
    # The classifier is skipped entirely for a top-tier override.
    assert decision.classification is None


def test_a_mid_tier_feature_floor_raises_but_does_not_cap(full_router):
    floored = full_router.route(
        request("Extract the account name.", feature="customer_summary")
    )
    assert floored.tier == 2
    assert "feature override" in floored.reason

    # A tier-3 prompt in the same feature still gets tier 3 -- a floor, not a cap.
    harder = full_router.route(
        request("Diagnose why this invoice is wrong.", feature="customer_summary")
    )
    assert harder.tier == 3


def test_a_risk_tag_raises_the_tier(full_router):
    decision = full_router.route(request("Extract the total.", risk_tags=["financial"]))

    assert decision.tier == 3


def test_several_risk_tags_still_reach_the_top_tier(full_router):
    decision = full_router.route(
        request("Extract the total.", risk_tags=["privacy", "legal"])
    )

    assert decision.tier == 3


def test_an_unrecognised_risk_tag_still_escalates(full_router):
    """Fail safe: a misspelled tag must not downgrade a sensitive request.

    A caller who writes 'legaal' has still signalled that this request matters.
    Ignoring unknown tags would send it to the cheapest model on a typo.
    """
    decision = full_router.route(
        request("Extract the invoice number.", risk_tags=["legaal"])
    )

    assert decision.tier == 3


# --- model selection ---------------------------------------------------------


def test_preference_order_within_a_tier_is_respected(full_router):
    decision = full_router.route(request("Extract the invoice number."))

    assert decision.model.name == "gpt-4o-mini"  # first tier-1 entry in routing.yaml


def test_mock_models_serve_when_no_credentials_are_configured(mock_only_router):
    """This is what makes the repo runnable straight after a clone."""
    decision = mock_only_router.route(request("Extract the invoice number."))

    assert decision.model.name == "mock-cheap"
    assert decision.model.provider == "mock"


def test_an_unreachable_provider_is_skipped(policy):
    router = Router(
        ModelRegistry.load(), policy, available_providers={"anthropic", "mock"}
    )

    decision = router.route(request("Extract the invoice number."))

    assert decision.model.name == "claude-haiku-4-5"  # gpt-4o-mini was skipped


def test_a_prompt_too_large_for_a_tier_upgrades_the_model(policy):
    """Better a stronger model than a truncated prompt."""
    router = Router(
        ModelRegistry.load(), policy, available_providers={"openai", "anthropic"}
    )

    decision = router.route(request("Extract the total."), input_tokens=300_000)

    # Every tier-1 and tier-2 OpenAI model caps at 128k; Sonnet 5 handles 1M.
    assert decision.model.max_context_tokens >= 300_000
    assert "upgraded" in decision.reason or decision.tier >= 2


def test_no_reachable_provider_is_a_clear_error(policy):
    router = Router(ModelRegistry.load(), policy, available_providers=set())

    with pytest.raises(RoutingError, match="no model available"):
        router.route(request())


# --- caller preference -------------------------------------------------------


def test_an_explicit_model_preference_is_honoured(full_router):
    decision = full_router.route(
        request("Extract the total.", preferred_model="claude-opus-5")
    )

    assert decision.model.name == "claude-opus-5"
    assert "explicitly" in decision.reason


def test_an_unknown_preferred_model_is_rejected(full_router):
    with pytest.raises(RegistryError):
        full_router.route(request(preferred_model="gpt-9-ultra"))


def test_a_preferred_model_on_an_unreachable_provider_fails_loudly(mock_only_router):
    """Silently substituting a different model would hide the substitution."""
    with pytest.raises(RoutingError, match="not configured"):
        mock_only_router.route(request(preferred_model="claude-opus-5"))


def test_a_preferred_model_that_cannot_fit_the_prompt_fails_loudly(full_router):
    with pytest.raises(RoutingError, match="context window"):
        full_router.route(
            request(preferred_model="claude-haiku-4-5"), input_tokens=500_000
        )


# --- policy loading ----------------------------------------------------------


def test_the_shipped_routing_config_covers_all_tiers(policy):
    for tier in (1, 2, 3):
        assert policy.tiers[tier]

    assert policy.feature_overrides["payment_dispute"] == 3
    assert policy.risk_tag_overrides["legal"] == 3


def test_every_model_named_in_routing_config_exists(policy):
    """Guards against a typo in routing.yaml surfacing as a runtime error."""
    registry = ModelRegistry.load()

    for models in policy.tiers.values():
        for name in models:
            assert registry.get(name)


def test_a_missing_routing_file_is_reported_clearly(tmp_path):
    with pytest.raises(RoutingError, match="not found"):
        RoutingPolicy.load(tmp_path / "nope.yaml")


def test_a_tier_with_no_models_is_rejected(tmp_path):
    path = tmp_path / "routing.yaml"
    path.write_text("tiers:\n  1: [mock-cheap]\n  2: []\n  3: [mock-strong]\n")

    with pytest.raises(RoutingError, match="tier 2"):
        RoutingPolicy.load(path)
