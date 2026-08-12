"""The HTTP layer.

Three endpoints, and no business logic: this module maps requests onto the
gateway and exceptions onto status codes. The interesting behaviour lives in
``app/gateway.py`` and ``app/budget/``.

Every error response is an :class:`~app.schemas.ErrorDetail` -- what happened,
and what to do about it. A caller who gets a 402 should not have to read our
source to find out which budget ran out.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.budget.counters import SpendCounters
from app.budget.enforcer import BudgetEnforcer
from app.db.engine import create_tables, get_engine, session_scope
from app.db.repository import Repository
from app.dependencies import (
    get_budget_policies,
    get_counters,
    get_enforcer,
    get_gateway,
    get_providers,
    get_registry,
    get_repository,
)
from app.gateway import BudgetExceeded, Gateway
from app.providers.base import ProviderError
from app.registry import ModelRegistry, RegistryError
from app.routing.router import PreferenceError, RoutingError
from app.schemas import BudgetView, ChatRequest, ChatResponse, ErrorDetail
from app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables, then restore budget counters from the usage log.

    The seeding step matters more than it looks. Counters are the fast view of
    what a team has spent; the database is the durable one. Starting up without
    reconciling them would hand every team a fresh budget on each deploy.
    """
    create_tables()

    enforcer = get_enforcer()
    with session_scope() as session:
        repo = Repository(session)
        for team_id in get_budget_policies().known_teams():
            await enforcer.seed(
                team_id=team_id,
                daily_spend_usd=repo.spend_today(team_id),
                monthly_spend_usd=repo.spend_this_month(team_id),
            )

    yield

    await get_counters().close()


app = FastAPI(
    title="LLM Spend Control Center",
    version="0.1.0",
    summary="Routes LLM traffic to the cheapest capable model and enforces budgets.",
    lifespan=lifespan,
)


# --- endpoints ---------------------------------------------------------------


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    gateway: Gateway = Depends(get_gateway),
    repo: Repository = Depends(get_repository),
) -> ChatResponse:
    """Route, budget-check, and run one chat completion.

    The response reports the model chosen, why it was chosen, what it cost, and
    what it would have cost on the baseline model -- so a single call is enough
    to see the routing decision without reading the logs.
    """
    return await gateway.handle(request, repo)


@app.get("/v1/budgets/{team_id}", response_model=BudgetView)
async def budget(
    team_id: str,
    enforcer: BudgetEnforcer = Depends(get_enforcer),
) -> BudgetView:
    """Current spend against limits for one team."""
    snapshot = await enforcer.snapshot(team_id)
    daily_percent = snapshot["daily_percent_used"]
    monthly_percent = snapshot["monthly_percent_used"]
    threshold = get_settings().budget_warn_threshold * 100

    if daily_percent > 100 or monthly_percent > 100:
        status = "blocked"
    elif max(daily_percent, monthly_percent) >= threshold:
        status = "warn"
    else:
        status = "allow"

    return BudgetView(team_id=team_id, status=status, **snapshot)


@app.get("/health")
async def health(
    counters: SpendCounters = Depends(get_counters),
    providers: dict[str, object] = Depends(get_providers),
    registry: ModelRegistry = Depends(get_registry),
) -> dict[str, object]:
    """Readiness: can we reach the database and the counters?

    Providers are reported as configured, not probed. Probing them on every
    health check would cost money and add latency to a call whose whole purpose
    is to be cheap.
    """
    checks: dict[str, object] = {}

    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # pragma: no cover - needs a broken database
        checks["database"] = f"error: {exc}"

    try:
        await counters.get(["health:probe"])
        checks["counters"] = get_settings().counters_backend()
    except Exception as exc:  # pragma: no cover - needs a broken Redis
        checks["counters"] = f"error: {exc}"

    checks["providers_configured"] = sorted(providers)
    checks["models_available"] = len(registry.all())
    checks["status"] = (
        "ok" if checks["database"] == "ok" and "error" not in str(checks["counters"])
        else "degraded"
    )
    return checks


# --- error mapping -----------------------------------------------------------


def _error(status_code: int, error: str, message: str, remedy: str | None = None):
    return JSONResponse(
        status_code=status_code,
        content=ErrorDetail(error=error, message=message, remedy=remedy).model_dump(),
    )


@app.exception_handler(BudgetExceeded)
async def handle_budget_exceeded(_: Request, exc: BudgetExceeded):
    """402: the budget is gone. Say which one, and how to get through anyway."""
    return _error(
        402,
        "budget_exceeded",
        exc.decision.reason,
        remedy=(
            "Raise the limit in config/budgets.yaml, wait for the period to reset, "
            "or resend with priority='high' to override (the override is recorded "
            "in the budget ledger)."
        ),
    )


@app.exception_handler(PreferenceError)
async def handle_preference_error(_: Request, exc: PreferenceError):
    """400: the caller asked for a model that cannot serve this request."""
    return _error(
        400,
        "model_unavailable",
        str(exc),
        remedy="Drop preferred_model to let the router choose, or pick a reachable model.",
    )


@app.exception_handler(RoutingError)
async def handle_routing_error(_: Request, exc: RoutingError):
    """503: nothing we can reach is able to serve this."""
    return _error(
        503,
        "no_model_available",
        str(exc),
        remedy="Configure a provider API key, or enable Ollama, and retry.",
    )


@app.exception_handler(RegistryError)
async def handle_registry_error(_: Request, exc: RegistryError):
    """422: the request names something that does not exist."""
    return _error(422, "unknown_model", str(exc))


@app.exception_handler(ProviderError)
async def handle_provider_error(_: Request, exc: ProviderError):
    """503: upstream failed. The budget reservation has already been released."""
    return _error(
        503,
        "provider_error",
        str(exc),
        remedy="Retry. No budget was consumed -- the reservation was released.",
    )
