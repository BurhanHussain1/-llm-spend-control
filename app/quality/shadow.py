"""Offline verification of a cheap model's answer.

Runs after the user already has their response, so nothing here can slow a
request down or fail one. The job is to answer a question the savings number
cannot answer on its own: *was the cheap answer actually good enough?*

The sequence is the same discipline as the main pipeline, because a verification
call is a real provider call that costs real money:

    reserve -> call the stronger model -> settle -> log -> grade -> record

That cost is logged with ``kind="shadow"`` so the report can show routed spend
and verification overhead separately. Verification is not free, and a savings
figure that hides its own measurement cost is not an honest figure.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.budget.enforcer import BudgetEnforcer
from app.budget.estimator import estimate_cost
from app.db.engine import session_scope
from app.db.repository import Repository
from app.providers.base import Provider, ProviderError
from app.quality.judge import Judge, Verdict
from app.registry import Model, ModelRegistry
from app.schemas import Message


@dataclass(frozen=True)
class ShadowJob:
    """Everything a verification needs, captured before the request's session closes."""

    team_id: str
    feature: str
    priority: str
    messages: list[Message]
    candidate_text: str
    candidate_model: str
    usage_event_id: int
    max_output_tokens: int


class ShadowVerifier:
    """Compares a sampled answer against a stronger model's and records the result."""

    def __init__(
        self,
        registry: ModelRegistry,
        reference_model: Model,
        provider: Provider,
        enforcer: BudgetEnforcer,
        judge: Judge,
    ) -> None:
        self._registry = registry
        self._reference_model = reference_model
        self._provider = provider
        self._enforcer = enforcer
        self._judge = judge

    async def run(self, job: ShadowJob) -> Verdict | None:
        """Verify one answer. Returns None if verification could not be run.

        Never raises. This runs detached from the request, so an exception here
        would surface as an unexplained background error and change nothing for
        the user.
        """
        try:
            return await self._run(job)
        except Exception as exc:  # pragma: no cover - defensive
            self._record_skip(job, f"verification failed: {exc}")
            return None

    async def _run(self, job: ShadowJob) -> Verdict | None:
        model = self._reference_model
        estimate = estimate_cost(job.messages, model, job.max_output_tokens)

        # A verification call spends money, so it goes through the budget like
        # anything else. It is deliberately sent at low priority: verifying our
        # own routing must never be the thing that exhausts a team's budget.
        budget = await self._enforcer.reserve(
            team_id=job.team_id,
            feature=job.feature,
            priority="low",
            estimated_cost_usd=estimate.cost_usd,
        )
        if not budget.allowed:
            self._record_skip(job, f"skipped: {budget.reason}")
            return None

        try:
            completion = await self._provider.complete(
                model=model.name,
                messages=job.messages,
                max_output_tokens=job.max_output_tokens,
            )
        except ProviderError as exc:
            await self._enforcer.release(
                job.team_id, job.feature, budget.reserved_usd
            )
            self._record_skip(job, f"reference model unavailable: {exc}")
            return None

        cost = model.cost(completion.input_tokens, completion.output_tokens)
        await self._enforcer.settle(
            team_id=job.team_id,
            feature=job.feature,
            reserved_usd=budget.reserved_usd,
            actual_cost_usd=cost,
        )

        prompt_text = "\n".join(message.content for message in job.messages)
        verdict = await self._judge.grade(
            prompt=prompt_text,
            candidate=job.candidate_text,
            reference=completion.text,
        )

        with session_scope() as session:
            repo = Repository(session)
            repo.log_usage(
                team_id=job.team_id,
                feature=job.feature,
                priority="low",
                model=model.name,
                provider=model.provider,
                tier=model.tier,
                routing_reason=f"shadow verification of {job.candidate_model}",
                status="ok",
                kind="shadow",
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                latency_ms=completion.latency_ms,
                cost_usd=cost,
                estimated_cost_usd=estimate.cost_usd,
                # A verification call has no counterfactual -- it *is* the
                # baseline model. Recording a baseline cost here would double
                # count it into the savings comparison.
                baseline_cost_usd=0.0,
                prompt=prompt_text,
            )
            repo.record_quality_check(
                prompt=prompt_text,
                chosen_model=job.candidate_model,
                better_model=model.name,
                passed=verdict.passed,
                score=verdict.score,
                reason=f"[{verdict.judge}] {verdict.reason}",
                usage_event_id=job.usage_event_id,
            )

        return verdict

    def _record_skip(self, job: ShadowJob, reason: str) -> None:
        """Record that verification did not happen, and why.

        Stored with ``passed=None``, not ``False``. A check that never ran is not
        evidence of a bad answer, and counting it as one would understate the
        pass rate for reasons unrelated to quality. Recording nothing at all would
        be worse still -- the skip would be invisible.
        """
        prompt_text = "\n".join(message.content for message in job.messages)
        with session_scope() as session:
            Repository(session).record_quality_check(
                prompt=prompt_text,
                chosen_model=job.candidate_model,
                better_model=self._reference_model.name,
                passed=None,
                score=None,
                reason=f"[not-run] {reason}",
                usage_event_id=job.usage_event_id,
            )
