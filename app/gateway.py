"""The request pipeline: classify, route, reserve, call, settle, log.

This is where the whole project comes together, in a fixed order:

    classify -> route -> estimate -> reserve -> call provider -> settle -> log

Two invariants hold on every path out of :meth:`Gateway.handle`:

1. **A reservation is always resolved.** Either it settles to the real cost, or it
   is released in full. A crashed provider call never leaves money held against a
   team's budget.
2. **Every attempt is logged.** Success, budget block, and provider failure all
   write a ``usage_events`` row and matching ledger entries. That is what makes
   the block rate and the per-provider error rate reportable rather than guessed.

The gateway owns the database transaction, committing at each terminal point --
including the failure paths, since a rolled-back audit trail records nothing.

HTTP knows nothing about this module and this module knows nothing about HTTP;
``app/main.py`` maps the exceptions below onto status codes.
"""

from __future__ import annotations

from collections.abc import Callable

from app.budget.enforcer import BudgetDecision, BudgetEnforcer
from app.budget.estimator import estimate_cost
from app.db.repository import Repository
from app.providers.base import Completion, Provider, ProviderError
from app.quality.sampler import should_escalate, should_shadow_check
from app.quality.shadow import ShadowJob, ShadowVerifier
from app.registry import Model, ModelRegistry
from app.routing.router import Router
from app.schemas import ChatRequest, ChatResponse
from app.tokens import count_message_tokens

#: Signature of the callable used to run verification after responding. FastAPI's
#: ``BackgroundTasks.add_task`` matches it, which keeps HTTP out of this module.
Scheduler = Callable[..., None]


class BudgetExceeded(Exception):
    """A request was refused because a budget is exhausted.

    Carries the decision so the API can tell the caller which limit broke, what
    the current spend is, and how to proceed.
    """

    def __init__(self, decision: BudgetDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class Gateway:
    """Runs one chat request through the full pipeline."""

    def __init__(
        self,
        registry: ModelRegistry,
        router: Router,
        enforcer: BudgetEnforcer,
        providers: dict[str, Provider],
        verifier: ShadowVerifier | None = None,
        shadow_sample_rate: float = 0.0,
        escalate_high_priority: bool = True,
    ) -> None:
        self._registry = registry
        self._router = router
        self._enforcer = enforcer
        self._providers = providers
        self._verifier = verifier
        self._shadow_sample_rate = shadow_sample_rate
        self._escalate_high_priority = escalate_high_priority

    async def handle(
        self,
        request: ChatRequest,
        repo: Repository,
        schedule: Scheduler | None = None,
    ) -> ChatResponse:
        # 1-2. Classify and route. Token count comes first so the router can
        #      reject models whose context window is too small.
        prompt_tokens = count_message_tokens(
            [{"role": m.role, "content": m.content} for m in request.messages]
        )
        routing = self._router.route(request, input_tokens=prompt_tokens)
        model = routing.model

        # 3. Estimate what it will cost on the model we actually chose.
        estimate = estimate_cost(request.messages, model, request.max_output_tokens)
        baseline = self._registry.baseline()

        # 4. Reserve before spending anything.
        budget = await self._enforcer.reserve(
            team_id=request.team_id,
            feature=request.feature,
            priority=request.priority,
            estimated_cost_usd=estimate.cost_usd,
        )

        if not budget.allowed:
            self._record_refusal(repo, request, routing, estimate, budget, baseline)
            raise BudgetExceeded(budget)

        # 5. Call the provider.
        provider = self._providers[model.provider]
        try:
            completion = await provider.complete(
                model=model.name,
                messages=request.messages,
                max_output_tokens=request.max_output_tokens,
            )
        except ProviderError as exc:
            await self._enforcer.release(
                request.team_id, request.feature, budget.reserved_usd
            )
            self._record_failure(
                repo, request, routing, estimate, budget, baseline, str(exc)
            )
            raise

        # 6. Settle: replace the estimate with the real cost.
        cost = model.cost(completion.input_tokens, completion.output_tokens)
        delta = await self._enforcer.settle(
            team_id=request.team_id,
            feature=request.feature,
            reserved_usd=budget.reserved_usd,
            actual_cost_usd=cost,
        )

        # The counterfactual: the same token counts priced on the baseline model.
        # Conservative on purpose -- a stronger model would likely have produced
        # *more* output tokens, so this understates the baseline and therefore
        # understates the savings.
        baseline_cost = baseline.cost(completion.input_tokens, completion.output_tokens)

        # 7. Log.
        event = repo.log_usage(
            team_id=request.team_id,
            feature=request.feature,
            priority=request.priority,
            model=model.name,
            provider=model.provider,
            tier=routing.tier,
            routing_reason=routing.reason,
            status="ok",
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
            cost_usd=cost,
            estimated_cost_usd=estimate.cost_usd,
            baseline_cost_usd=baseline_cost,
            prompt=request.prompt_text(),
        )
        self._ledger_reserve(repo, request, estimate.cost_usd, event.id)
        if budget.status == "override":
            repo.record_ledger(
                team_id=request.team_id,
                feature=request.feature,
                action="override",
                amount_usd=0.0,
                usage_event_id=event.id,
                note=budget.reason[:255],
            )
        repo.record_ledger(
            team_id=request.team_id,
            feature=request.feature,
            action="settle",
            amount_usd=delta,
            usage_event_id=event.id,
            note=f"estimate ${estimate.cost_usd:.6f} -> actual ${cost:.6f}",
        )
        repo.session.commit()

        # 8. Escalate if the answer looks unusable, before anyone sees it.
        escalated = await self._maybe_escalate(
            request, repo, completion, model, routing.reason, budget
        )
        if escalated is not None:
            return escalated

        # 9. Sample for offline verification. Runs after the response is sent, so
        #    it can neither slow this request down nor fail it.
        self._maybe_schedule_verification(
            request, model, completion, event.id, schedule
        )

        return ChatResponse(
            text=completion.text,
            chosen_model=model.name,
            provider=model.provider,
            tier=routing.tier,
            routing_reason=routing.reason,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=cost,
            baseline_cost_usd=baseline_cost,
            latency_ms=completion.latency_ms,
            budget_status=budget.status,
            warnings=budget.warnings,
        )

    # --- escalation ----------------------------------------------------------

    def _stronger_models(self, tier: int) -> list[Model]:
        """Reachable models above `tier`, so escalation has somewhere to go."""
        return [
            model
            for model in self._registry.all()
            if model.tier > tier and model.provider in self._providers
        ]

    async def _maybe_escalate(
        self,
        request: ChatRequest,
        repo: Repository,
        completion: Completion,
        model: Model,
        original_reason: str,
        primary_budget: BudgetDecision,
    ) -> ChatResponse | None:
        """Rerun on a stronger model when the cheap answer is not usable.

        Returns the replacement response, or None to keep the original.

        The rerun is budgeted like any other call. If the budget refuses it, the
        original answer is returned with a warning rather than an error -- a
        usable-if-imperfect answer beats no answer, and the warning makes the
        compromise visible instead of silent.
        """
        candidates = self._stronger_models(model.tier)
        strongest_tier = max((m.tier for m in candidates), default=model.tier)

        decision = should_escalate(
            request=request,
            completion=completion,
            model=model,
            strongest_tier=strongest_tier,
            escalate_high_priority=self._escalate_high_priority,
        )
        if not decision.needed or not candidates:
            return None

        # The primary call already broke a budget and was let through as an
        # override. Spending a second time on a *precaution* would double down
        # past a limit that has already failed. A broken answer is still worth
        # fixing, so only the priority-based escalation is suppressed here.
        if primary_budget.status == "override" and decision.category == "priority":
            return None

        reason = decision.reason

        stronger = self._registry.strongest(among=candidates)
        estimate = estimate_cost(request.messages, stronger, request.max_output_tokens)
        baseline = self._registry.baseline()

        budget = await self._enforcer.reserve(
            team_id=request.team_id,
            feature=request.feature,
            priority=request.priority,
            estimated_cost_usd=estimate.cost_usd,
        )
        if not budget.allowed:
            return None  # keep the original answer; the warning is added below

        try:
            retry = await self._providers[stronger.provider].complete(
                model=stronger.name,
                messages=request.messages,
                max_output_tokens=request.max_output_tokens,
            )
        except ProviderError:
            await self._enforcer.release(
                request.team_id, request.feature, budget.reserved_usd
            )
            return None  # the first answer is still better than an error

        cost = stronger.cost(retry.input_tokens, retry.output_tokens)
        delta = await self._enforcer.settle(
            team_id=request.team_id,
            feature=request.feature,
            reserved_usd=budget.reserved_usd,
            actual_cost_usd=cost,
        )
        escalation_reason = f"escalated to {stronger.name}: {reason}"

        event = repo.log_usage(
            team_id=request.team_id,
            feature=request.feature,
            priority=request.priority,
            model=stronger.name,
            provider=stronger.provider,
            tier=stronger.tier,
            routing_reason=escalation_reason,
            status="ok",
            kind="escalation",
            input_tokens=retry.input_tokens,
            output_tokens=retry.output_tokens,
            latency_ms=retry.latency_ms,
            cost_usd=cost,
            estimated_cost_usd=estimate.cost_usd,
            # No counterfactual on an escalation: this *is* effectively the
            # baseline model, and claiming savings on it would be double counting.
            baseline_cost_usd=cost,
            prompt=request.prompt_text(),
        )
        self._ledger_reserve(repo, request, estimate.cost_usd, event.id)
        repo.record_ledger(
            team_id=request.team_id,
            feature=request.feature,
            action="settle",
            amount_usd=delta,
            usage_event_id=event.id,
            note=escalation_reason[:255],
        )
        repo.session.commit()

        return ChatResponse(
            text=retry.text,
            chosen_model=stronger.name,
            provider=stronger.provider,
            tier=stronger.tier,
            routing_reason=f"{original_reason}; {escalation_reason}",
            input_tokens=retry.input_tokens,
            output_tokens=retry.output_tokens,
            cost_usd=cost,
            baseline_cost_usd=cost,
            latency_ms=retry.latency_ms,
            budget_status=budget.status,
            warnings=[*budget.warnings, f"first attempt discarded: {reason}"],
            escalated=True,
        )

    # --- shadow verification -------------------------------------------------

    def _maybe_schedule_verification(
        self,
        request: ChatRequest,
        model: Model,
        completion: Completion,
        usage_event_id: int,
        schedule: Scheduler | None,
    ) -> None:
        """Queue an offline quality check for a sampled fraction of answers."""
        if self._verifier is None or schedule is None:
            return

        candidates = self._stronger_models(model.tier)
        strongest_tier = max((m.tier for m in candidates), default=model.tier)

        if not should_shadow_check(
            prompt=request.prompt_text(),
            model=model,
            strongest_tier=strongest_tier,
            rate=self._shadow_sample_rate,
        ):
            return

        schedule(
            self._verifier.run,
            ShadowJob(
                team_id=request.team_id,
                feature=request.feature,
                priority=request.priority,
                messages=list(request.messages),
                candidate_text=completion.text,
                candidate_model=model.name,
                usage_event_id=usage_event_id,
                max_output_tokens=request.max_output_tokens,
            ),
        )

    # --- failure paths -------------------------------------------------------

    def _record_refusal(self, repo, request, routing, estimate, budget, baseline) -> None:
        """Log a budget block: a zero-cost attempt plus a reserve/release pair."""
        event = repo.log_usage(
            team_id=request.team_id,
            feature=request.feature,
            priority=request.priority,
            model=routing.model.name,
            provider=routing.model.provider,
            tier=routing.tier,
            routing_reason=routing.reason,
            status="blocked",
            estimated_cost_usd=estimate.cost_usd,
            baseline_cost_usd=baseline.cost(estimate.input_tokens, estimate.output_tokens),
            prompt=request.prompt_text(),
            error=budget.reason,
        )
        self._ledger_reserve(repo, request, estimate.cost_usd, event.id)
        repo.record_ledger(
            team_id=request.team_id,
            feature=request.feature,
            action="release",
            amount_usd=-estimate.cost_usd,
            usage_event_id=event.id,
            note=f"blocked: {budget.reason}"[:255],
        )
        repo.session.commit()

    def _record_failure(
        self, repo, request, routing, estimate, budget, baseline, error: str
    ) -> None:
        """Log a provider failure and the release that gave the budget back."""
        event = repo.log_usage(
            team_id=request.team_id,
            feature=request.feature,
            priority=request.priority,
            model=routing.model.name,
            provider=routing.model.provider,
            tier=routing.tier,
            routing_reason=routing.reason,
            status="provider_error",
            estimated_cost_usd=estimate.cost_usd,
            baseline_cost_usd=baseline.cost(estimate.input_tokens, estimate.output_tokens),
            prompt=request.prompt_text(),
            error=error,
        )
        self._ledger_reserve(repo, request, budget.reserved_usd, event.id)
        repo.record_ledger(
            team_id=request.team_id,
            feature=request.feature,
            action="release",
            amount_usd=-budget.reserved_usd,
            usage_event_id=event.id,
            note=f"provider error: {error}"[:255],
        )
        repo.session.commit()

    def _ledger_reserve(
        self, repo: Repository, request: ChatRequest, amount: float, event_id: int
    ) -> None:
        repo.record_ledger(
            team_id=request.team_id,
            feature=request.feature,
            action="reserve",
            amount_usd=amount,
            usage_event_id=event_id,
            note="pre-call estimate",
        )
