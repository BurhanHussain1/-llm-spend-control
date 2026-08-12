"""Object construction and wiring.

Everything the gateway needs is built exactly once per process and handed to
request handlers through FastAPI's dependency system. Keeping it here means
``main.py`` stays an HTTP layer and the components stay constructible in tests
without a running server.

Which providers exist is decided by configuration, not by code: the paid
providers appear when their API key is set, Ollama when it is explicitly enabled,
and the mock provider always. That last part is why a fresh clone can serve
traffic with no credentials at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from app.budget.counters import SpendCounters, build_counters
from app.budget.enforcer import BudgetEnforcer
from app.budget.policies import BudgetPolicies
from app.db.engine import get_session_factory
from app.db.repository import Repository
from app.gateway import Gateway
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import Provider
from app.providers.mock import MockProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider
from app.registry import ModelRegistry
from app.routing.router import Router, RoutingPolicy
from app.settings import get_settings


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry.load()


@lru_cache
def get_budget_policies() -> BudgetPolicies:
    return BudgetPolicies.load()


@lru_cache
def get_routing_policy() -> RoutingPolicy:
    return RoutingPolicy.load()


@lru_cache
def get_counters() -> SpendCounters:
    return build_counters(get_settings().redis_url)


@lru_cache
def get_enforcer() -> BudgetEnforcer:
    return BudgetEnforcer(
        policies=get_budget_policies(),
        counters=get_counters(),
        warn_threshold=get_settings().budget_warn_threshold,
    )


@lru_cache
def get_providers() -> dict[str, Provider]:
    """Build the providers this deployment can actually reach."""
    settings = get_settings()
    registry = get_registry()

    # Always present: it needs no credentials and costs nothing, which is what
    # lets the test suite and the workload simulation run offline.
    providers: dict[str, Provider] = {"mock": MockProvider(registry)}

    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(
            api_key=settings.openai_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if settings.enable_ollama:
        providers["ollama"] = OllamaProvider(
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.request_timeout_seconds,
        )
    return providers


@lru_cache
def get_router() -> Router:
    return Router(
        registry=get_registry(),
        policy=get_routing_policy(),
        available_providers=set(get_providers()),
    )


@lru_cache
def get_gateway() -> Gateway:
    return Gateway(
        registry=get_registry(),
        router=get_router(),
        enforcer=get_enforcer(),
        providers=get_providers(),
    )


def get_repository() -> Iterator[Repository]:
    """One database session per request.

    The gateway commits at its own terminal points, so this only guarantees the
    session is closed and that an unhandled error cannot leave a half-written
    transaction behind.
    """
    session = get_session_factory()()
    try:
        yield Repository(session)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_dependencies() -> None:
    """Drop every cached singleton so new settings take effect. For tests."""
    for builder in (
        get_registry,
        get_budget_policies,
        get_routing_policy,
        get_counters,
        get_enforcer,
        get_providers,
        get_router,
        get_gateway,
    ):
        builder.cache_clear()
