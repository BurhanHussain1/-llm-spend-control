"""End-to-end tests through the HTTP layer.

These run against the mock provider on a throwaway SQLite database, so the whole
pipeline -- classify, route, reserve, call, settle, log -- is exercised with no
credentials and no network.
"""

import pytest
from fastapi.testclient import TestClient

from app.budget.counters import InMemoryCounters
from app.budget.enforcer import BudgetEnforcer
from app.budget.policies import BudgetPolicies
from app.db.engine import create_tables, get_session_factory, reset_engine
from app.db.repository import Repository
from app.dependencies import get_enforcer, get_gateway, reset_dependencies
from app.gateway import Gateway
from app.providers.base import ProviderError
from app.providers.mock import MockProvider
from app.registry import ModelRegistry
from app.routing.router import Router, RoutingPolicy
from app.settings import get_settings

TIGHT_POLICY = """
defaults:
  daily_limit_usd: 10.00
  monthly_limit_usd: 100.00
teams:
  search:
    daily_limit_usd: 25.00
    monthly_limit_usd: 500.00
  tiny:
    daily_limit_usd: 0.0001
    monthly_limit_usd: 0.0002
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a fresh database and mock-only providers."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    get_settings.cache_clear()
    reset_engine()
    reset_dependencies()

    policies_path = tmp_path / "budgets.yaml"
    policies_path.write_text(TIGHT_POLICY, encoding="utf-8")
    enforcer = BudgetEnforcer(
        policies=BudgetPolicies.load(policies_path),
        counters=InMemoryCounters(),
        warn_threshold=0.8,
    )

    from app.main import app

    # The gateway resolves its own collaborators, so overriding the enforcer
    # alone would only affect the budgets endpoint -- the pipeline would still
    # use the real config/budgets.yaml limits. Override both.
    registry = ModelRegistry.load()
    gateway = Gateway(
        registry=registry,
        router=Router(
            registry, RoutingPolicy.load(), available_providers={"mock"}
        ),
        enforcer=enforcer,
        providers={"mock": MockProvider(registry)},
    )
    app.dependency_overrides[get_enforcer] = lambda: enforcer
    app.dependency_overrides[get_gateway] = lambda: gateway

    with TestClient(app) as test_client:
        test_client.enforcer = enforcer  # exposed so tests can inspect counters
        yield test_client

    app.dependency_overrides.clear()
    reset_engine()
    reset_dependencies()
    get_settings.cache_clear()


def read_events(tmp_path_factory=None):
    """Read usage rows straight from the database the app just wrote to."""
    session = get_session_factory()()
    try:
        return Repository(session).recent_events()
    finally:
        session.close()


def payload(**overrides):
    body = {
        "messages": [{"role": "user", "content": "Extract the invoice number."}],
        "team_id": "search",
        "feature": "summaries",
    }
    body.update(overrides)
    return body


# --- happy path --------------------------------------------------------------


def test_a_request_is_routed_priced_and_answered(client):
    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 200
    body = response.json()

    assert body["text"]
    assert body["provider"] == "mock"  # no credentials configured
    assert body["tier"] == 1  # an extraction prompt
    assert body["cost_usd"] > 0
    assert body["budget_status"] == "allow"
    assert body["latency_ms"] > 0


def test_the_response_explains_the_routing_decision(client):
    """A single call should be enough to understand why this model was chosen."""
    body = client.post("/v1/chat", json=payload()).json()

    assert "classifier" in body["routing_reason"]
    assert "extraction verb" in body["routing_reason"]


def test_the_response_carries_the_baseline_counterfactual(client):
    """Savings have to be visible per request, not only in aggregate."""
    body = client.post("/v1/chat", json=payload()).json()

    assert body["baseline_cost_usd"] > body["cost_usd"]


def test_a_successful_request_writes_one_usage_row(client):
    client.post("/v1/chat", json=payload())

    events = read_events()
    assert len(events) == 1
    assert events[0].status == "ok"
    assert events[0].cost_usd > 0
    assert events[0].estimated_cost_usd > 0


def test_a_successful_request_leaves_a_reserve_and_settle_trail(client):
    """The ledger is what makes 'we reserved, then reconciled' auditable."""
    client.post("/v1/chat", json=payload())

    session = get_session_factory()()
    try:
        repo = Repository(session)
        event = repo.recent_events()[0]
        trail = repo.ledger_for_event(event.id)

        assert [entry.action for entry in trail] == ["reserve", "settle"]
        # Net ledger movement equals the real cost.
        assert round(sum(e.amount_usd for e in trail), 8) == round(event.cost_usd, 8)
    finally:
        session.close()


def test_a_hard_prompt_routes_to_a_stronger_model(client):
    body = client.post(
        "/v1/chat",
        json=payload(messages=[{"role": "user", "content": "Diagnose why this deployment rolled back."}]),
    ).json()

    assert body["tier"] == 3
    assert body["chosen_model"] == "mock-strong"


def test_a_risk_tag_forces_a_strong_model(client):
    body = client.post(
        "/v1/chat", json=payload(risk_tags=["legal"])
    ).json()

    assert body["tier"] == 3
    assert "risk tag" in body["routing_reason"]


def test_a_feature_override_forces_a_strong_model(client):
    body = client.post(
        "/v1/chat", json=payload(feature="payment_dispute")
    ).json()

    assert body["tier"] == 3
    assert "feature override" in body["routing_reason"]


# --- budget refusal ----------------------------------------------------------


def test_an_exhausted_budget_returns_402_and_says_why(client):
    response = client.post("/v1/chat", json=payload(team_id="tiny"))

    assert response.status_code == 402
    body = response.json()
    assert body["error"] == "budget_exceeded"
    assert "daily budget" in body["message"]
    assert "priority" in body["remedy"]  # tells the caller how to proceed


def test_a_blocked_request_is_logged_at_zero_cost(client):
    client.post("/v1/chat", json=payload(team_id="tiny"))

    events = read_events()
    assert len(events) == 1
    assert events[0].status == "blocked"
    assert events[0].cost_usd == 0.0
    assert events[0].error  # carries the reason


def test_a_blocked_request_consumes_no_budget(client):
    """Otherwise a flood of rejections would exhaust a budget that had room."""
    client.post("/v1/chat", json=payload(team_id="tiny"))

    snapshot = pytest_run(client.enforcer.snapshot("tiny"))
    assert snapshot["daily_spend_usd"] == pytest.approx(0.0)


def test_high_priority_overrides_an_exhausted_budget(client):
    blocked = client.post("/v1/chat", json=payload(team_id="tiny"))
    assert blocked.status_code == 402

    override = client.post("/v1/chat", json=payload(team_id="tiny", priority="high"))

    assert override.status_code == 200
    assert override.json()["budget_status"] == "override"
    assert override.json()["warnings"]


def test_an_override_is_recorded_in_the_ledger(client):
    """An override that left no trace would be indistinguishable from a bug."""
    client.post("/v1/chat", json=payload(team_id="tiny", priority="high"))

    session = get_session_factory()()
    try:
        repo = Repository(session)
        primary = next(
            e for e in repo.recent_events() if e.status == "ok" and e.kind == "primary"
        )
        actions = [entry.action for entry in repo.ledger_for_event(primary.id)]
        assert "override" in actions
    finally:
        session.close()


def test_an_override_does_not_also_trigger_an_escalation(client):
    """Overriding a blown budget once is a decision; twice is a bug.

    A high-priority request that punches through an exhausted budget must not
    then spend a second time on a precautionary rerun. A *broken* answer would
    still be escalated -- only the priority-based rerun is suppressed.
    """
    client.post("/v1/chat", json=payload(team_id="tiny", priority="high"))

    events = read_events()
    assert [e.kind for e in events] == ["primary"]


# --- provider failure --------------------------------------------------------


def test_a_provider_failure_returns_503_and_refunds_the_reservation(client, tmp_path):
    """The invariant that matters: an outage must not consume budget."""

    class BrokenProvider:
        name = "mock"

        async def complete(self, model, messages, max_output_tokens):
            raise ProviderError("mock", "simulated upstream outage")

    from app.main import app

    broken_gateway = Gateway(
        registry=ModelRegistry.load(),
        router=Router(
            ModelRegistry.load(), RoutingPolicy.load(), available_providers={"mock"}
        ),
        enforcer=client.enforcer,
        providers={"mock": BrokenProvider()},
    )
    app.dependency_overrides[get_gateway] = lambda: broken_gateway

    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 503
    assert response.json()["error"] == "provider_error"
    assert "No budget was consumed" in response.json()["remedy"]

    snapshot = pytest_run(client.enforcer.snapshot("search"))
    assert snapshot["daily_spend_usd"] == pytest.approx(0.0)

    events = read_events()
    assert events[0].status == "provider_error"
    assert "outage" in events[0].error


# --- caller preference and validation ---------------------------------------


def test_an_explicit_model_preference_is_honoured(client):
    body = client.post(
        "/v1/chat", json=payload(preferred_model="mock-strong")
    ).json()

    assert body["chosen_model"] == "mock-strong"
    assert "explicitly" in body["routing_reason"]


def test_an_unreachable_preferred_model_returns_400(client):
    response = client.post("/v1/chat", json=payload(preferred_model="claude-opus-5"))

    assert response.status_code == 400
    assert response.json()["error"] == "model_unavailable"
    assert "not configured" in response.json()["message"]


def test_an_unknown_preferred_model_returns_422(client):
    response = client.post("/v1/chat", json=payload(preferred_model="gpt-9-ultra"))

    assert response.status_code == 422
    assert response.json()["error"] == "unknown_model"


def test_an_empty_message_list_is_rejected(client):
    assert client.post("/v1/chat", json=payload(messages=[])).status_code == 422


def test_a_missing_team_id_is_rejected(client):
    body = payload()
    del body["team_id"]

    assert client.post("/v1/chat", json=body).status_code == 422


# --- budgets and health ------------------------------------------------------


def test_the_budget_endpoint_reports_spend_against_limits(client):
    client.post("/v1/chat", json=payload())

    body = client.get("/v1/budgets/search").json()

    assert body["team_id"] == "search"
    assert body["daily_spend_usd"] > 0
    assert body["daily_limit_usd"] == 25.00
    assert body["status"] == "allow"


def test_the_budget_endpoint_flags_an_exhausted_team(client):
    client.post("/v1/chat", json=payload(team_id="tiny", priority="high"))

    body = client.get("/v1/budgets/tiny").json()

    assert body["status"] == "blocked"
    assert body["daily_percent_used"] > 100


def test_health_reports_backends_without_probing_providers(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["counters"] == "memory"
    assert body["providers_configured"] == ["mock"]


def pytest_run(coroutine):
    """Run a coroutine from a synchronous test."""
    import asyncio

    return asyncio.run(coroutine)
