"""Push a mixed workload through the gateway and write a savings report.

Run it:

    python -m scripts.simulate_workload            # 1,000 requests
    python -m scripts.simulate_workload --count 200

Everything runs on the mock provider against a throwaway database in
``reports/``, so the run costs nothing, needs no credentials, and never touches
``data/spend.db``.

**What this is and is not.** The prompts come from this repo's own hand-labeled
corpus (``eval/``), not from a public dataset, so the tier mix is one we chose.
That is a real limitation and it is repeated in the report: the *mechanism* --
classify, route, reserve, settle, verify -- is what the numbers demonstrate. The
savings percentage would move on a different traffic mix.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# Point the whole app at a throwaway database before anything imports the engine.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SIM_DB = REPORTS_DIR / "simulation.db"
os.environ["DATABASE_URL"] = f"sqlite:///{SIM_DB.as_posix()}"
os.environ.pop("REDIS_URL", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

from app import reporting  # noqa: E402
from app.budget.counters import InMemoryCounters  # noqa: E402
from app.budget.enforcer import BudgetEnforcer  # noqa: E402
from app.budget.policies import BudgetPolicies  # noqa: E402
from app.db.engine import create_tables, get_session_factory, reset_engine  # noqa: E402
from app.db.models import UsageEvent, utcnow  # noqa: E402
from app.db.repository import Repository  # noqa: E402
from app.gateway import BudgetExceeded, Gateway  # noqa: E402
from app.providers.base import ProviderError  # noqa: E402
from app.providers.mock import MockProvider  # noqa: E402
from app.quality.judge import MechanicalJudge  # noqa: E402
from app.quality.shadow import ShadowVerifier  # noqa: E402
from app.registry import ModelRegistry  # noqa: E402
from app.routing.router import Router, RoutingError, RoutingPolicy  # noqa: E402
from app.schemas import ChatRequest, Message  # noqa: E402
from app.settings import get_settings  # noqa: E402

#: Teams and the features they call, mirroring config/budgets.yaml.
TRAFFIC = [
    ("search", "autocomplete"),
    ("search", "summaries"),
    ("support", "ticket_triage"),
    ("support", "customer_summary"),
    ("billing", "refund_decision"),
    ("billing", "invoice_extraction"),
    ("analytics", "report_summary"),
    ("analytics", "incident_postmortem"),
]

#: Priority mix: mostly normal, a little urgent, some batch work.
PRIORITIES = ["normal"] * 7 + ["low"] * 2 + ["high"]

#: Days to spread the log over, so the daily chart has more than one point.
SPREAD_DAYS = 14

SEED = 20260812


def load_prompts() -> list[dict]:
    """Every labeled prompt in the repo, calibration and holdout together."""
    prompts = []
    for name in ("labeled_prompts.jsonl", "holdout_prompts.jsonl"):
        path = PROJECT_ROOT / "eval" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def build_requests(count: int) -> list[ChatRequest]:
    """Compose `count` requests by cycling the corpus across teams and features."""
    corpus = load_prompts()
    rng = random.Random(SEED)  # seeded, so the report is reproducible
    requests = []

    for index in range(count):
        record = corpus[index % len(corpus)]
        team, feature = TRAFFIC[index % len(TRAFFIC)]

        # Vary the text slightly so sampling and caching are not degenerate.
        content = record["prompt"]
        if index >= len(corpus):
            content = f"{content} (case {index})"

        requests.append(
            ChatRequest(
                messages=[Message(role="user", content=content)],
                team_id=team,
                feature=feature,
                priority=rng.choice(PRIORITIES),
                risk_tags=list(record.get("risk_tags", [])),
            )
        )
    return requests


def build_gateway() -> tuple[Gateway, BudgetEnforcer, list]:
    registry = ModelRegistry.load()
    providers = {"mock": MockProvider(registry)}
    enforcer = BudgetEnforcer(
        policies=BudgetPolicies.load(),
        counters=InMemoryCounters(),
        warn_threshold=get_settings().budget_warn_threshold,
    )
    reference = registry.get("mock-strong")
    verifier = ShadowVerifier(
        registry=registry,
        reference_model=reference,
        provider=providers["mock"],
        enforcer=enforcer,
        judge=MechanicalJudge(),
    )
    scheduled: list = []

    gateway = Gateway(
        registry=registry,
        router=Router(registry, RoutingPolicy.load(), available_providers={"mock"}),
        enforcer=enforcer,
        providers=providers,
        verifier=verifier,
        shadow_sample_rate=get_settings().shadow_sample_rate,
        escalate_high_priority=get_settings().escalate_on_high_priority,
    )
    return gateway, enforcer, scheduled


async def run(count: int) -> None:
    if SIM_DB.exists():
        SIM_DB.unlink()  # a fresh run must not accumulate on the last one
    reset_engine()
    create_tables()

    gateway, _enforcer, scheduled = build_gateway()
    requests = build_requests(count)

    outcomes = {"ok": 0, "blocked": 0, "error": 0, "unroutable": 0}
    session_factory = get_session_factory()

    print(f"Running {count} requests through the gateway (mock provider)...")
    for index, request in enumerate(requests, start=1):
        session = session_factory()
        try:
            await gateway.handle(
                request, Repository(session), schedule=lambda fn, job: scheduled.append((fn, job))
            )
            outcomes["ok"] += 1
        except BudgetExceeded:
            outcomes["blocked"] += 1
        except ProviderError:
            outcomes["error"] += 1
        except RoutingError:
            outcomes["unroutable"] += 1
        finally:
            session.close()

        if index % 200 == 0:
            print(f"  {index}/{count}")

    print(f"Running {len(scheduled)} sampled quality checks...")
    for fn, job in scheduled:
        await fn(job)

    spread_events_over_days()
    write_report(outcomes, count)


def spread_events_over_days() -> None:
    """Backdate the log across `SPREAD_DAYS` so the daily chart has a shape.

    Presentation only: it changes when rows *say* they happened, not what they
    cost. Budgets were enforced against a single logical "now" during the run,
    which is why this is a fixup rather than a genuine multi-day simulation.
    """
    session = get_session_factory()()
    try:
        events = session.query(UsageEvent).order_by(UsageEvent.id).all()
        if not events:
            return
        start = utcnow() - timedelta(days=SPREAD_DAYS - 1)
        for index, event in enumerate(events):
            day = index * SPREAD_DAYS // len(events)
            event.created_at = start + timedelta(days=day, minutes=index % 1440)
        session.commit()
    finally:
        session.close()


def write_report(outcomes: dict[str, int], requested: int) -> None:
    session = get_session_factory()()
    try:
        repo = Repository(session)
        savings = reporting.savings_summary(repo)
        verification = reporting.verification_summary(repo)
        estimate = reporting.estimate_error(repo)
        tiers = repo.tier_distribution()
        latency = reporting.latency_percentiles(repo)

        report = REPORTS_DIR / "savings_report.md"
        report.write_text(
            _render(
                outcomes, requested, savings, verification, estimate, tiers, latency, repo
            ),
            encoding="utf-8",
        )

        csv_path = REPORTS_DIR / "usage.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["kind", "team", "feature", "priority", "model", "tier", "status",
                 "input_tokens", "output_tokens", "latency_ms", "cost_usd",
                 "estimated_cost_usd", "baseline_cost_usd", "routing_reason"]
            )
            for event in session.query(UsageEvent).order_by(UsageEvent.id):
                writer.writerow(
                    [event.kind, event.team_id, event.feature, event.priority,
                     event.model, event.tier, event.status, event.input_tokens,
                     event.output_tokens, event.latency_ms, event.cost_usd,
                     event.estimated_cost_usd, event.baseline_cost_usd,
                     event.routing_reason]
                )
    finally:
        session.close()

    print(f"\nWrote {report.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"\n  net saving      {savings.net_savings_percent:.1f}%")
    print(f"  gross saving    {savings.gross_savings_percent:.1f}%")
    print(f"  verifier pass   {verification.pass_rate_percent:.1f}% of {verification.graded} graded")
    print(f"  escalation rate {reporting.escalation_rate(repo):.1f}%")


def _render(outcomes, requested, savings, verification, estimate, tiers, latency, repo) -> str:
    tier_lines = "\n".join(
        f"| Tier {tier} | {count} | {count / max(sum(tiers.values()), 1):.1%} |"
        for tier, count in sorted(tiers.items())
    )
    latency_lines = "\n".join(
        f"| {row['model']} | {row['requests']} | {row['p50_ms']:.0f} | {row['p95_ms']:.0f} |"
        for row in latency
    )
    model_lines = "\n".join(
        f"| {name} | {count} | ${cost:.6f} |" for name, count, cost in repo.spend_by("model")
    )

    return f"""# Simulated workload -- savings report

Generated by `python -m scripts.simulate_workload`. All figures come from
`app/reporting.py`, the same code the dashboard uses.

## Headline

| Metric | Value |
|---|---|
| Requests | {savings.requests:,} of {requested:,} attempted |
| Provider calls | {savings.provider_calls:,} |
| Baseline (everything on the strongest model) | ${savings.baseline_cost_usd:.4f} |
| Routed spend | ${savings.routed_cost_usd:.4f} |
| Escalation spend | ${savings.escalation_cost_usd:.4f} |
| Verification spend | ${savings.verification_cost_usd:.4f} |
| **Total spend** | **${savings.total_spend_usd:.4f}** |
| Gross saving | ${savings.gross_savings_usd:.4f} ({savings.gross_savings_percent:.1f}%) |
| **Net saving** | **${savings.net_savings_usd:.4f} ({savings.net_savings_percent:.1f}%)** |
| Overhead on routed spend | {savings.overhead_percent_of_routed:.1f}% |

Gross saving ignores what the routing costs to operate. Net subtracts escalation
and verification. **Quote net.**

## Outcomes

| Outcome | Count |
|---|---|
| Answered | {outcomes['ok']:,} |
| Blocked by budget | {outcomes['blocked']:,} |
| Provider error | {outcomes['error']:,} |
| Unroutable | {outcomes['unroutable']:,} |

## Routing

{tier_lines or '| (no data) | | |'}

| Model | Calls | Cost |
|---|---|---|
{model_lines or '| (no data) | | |'}

Escalation rate: **{reporting.escalation_rate(repo):.1f}%** of routed requests were
rerun on a stronger model.

## Measured quality

| Metric | Value |
|---|---|
| Graded checks | {verification.graded:,} |
| Passed | {verification.passed:,} |
| Failed | {verification.failed:,} |
| Not run | {verification.not_run:,} |
| Pass rate | {verification.pass_rate_percent:.1f}% |

> {verification.caveat}

Checks that could not run are excluded from the pass rate rather than counted as
failures -- a check that never ran says nothing about the answer.

## Estimate accuracy

Budgets are enforced against a pre-call estimate, so this is the honesty check on
that enforcement.

| Metric | Value |
|---|---|
| Samples | {estimate['samples']:,} |
| Median absolute error | {estimate['median_absolute_percent']:.1f}% |
| Median signed error | {estimate['median_signed_percent']:+.1f}% |
| Mean absolute error | {estimate['mean_absolute_percent']:.1f}% |

Read the median. Percentage error divides by actual cost, so a single request that
produced almost no output can push the mean into the thousands of percent.

## Latency

| Model | Calls | p50 (ms) | p95 (ms) |
|---|---|---|---|
{latency_lines or '| (no data) | | | |'}

Latency is synthetic -- the mock provider reports plausible figures rather than
sleeping, so a 1,000-request run finishes in seconds.

## Limitations

1. **The prompt mix is ours.** Prompts come from this repo's hand-labeled corpus,
   not a public dataset, so the tier distribution -- and therefore the savings
   percentage -- reflects a mix we chose. The mechanism is what this
   demonstrates, not the specific percentage.
2. **The pass rate is mechanical.** With no provider credentials the judge checks
   for empty and truncated answers and cannot assess correctness. A real quality
   number needs an API key and an LLM judge.
3. **Mock pricing mirrors real pricing.** `mock-cheap`/`mid`/`strong` are priced
   identically to the Haiku/Sonnet/Opus tiers, so the cost ratios are realistic
   even though no provider was called.
4. **Budgets ran against one logical day.** Log timestamps are spread over
   {SPREAD_DAYS} days for charting, but enforcement during the run used a single
   "now", so the block count here is not a multi-day block rate.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(run(args.count))


if __name__ == "__main__":
    main()
