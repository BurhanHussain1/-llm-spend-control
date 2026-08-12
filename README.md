# LLM Spend Control Center

A routing and budgeting layer that sits in front of LLM calls. It tracks spend per
team and feature, routes each request to the cheapest model that can actually handle
it, and blocks requests before the money is spent when a budget runs out.

> **Result:** _pending — filled in from `reports/savings_report.md` in Phase 8._

**Status:** Phase 0 of 8 complete (repo skeleton and CI). See [`PLAN.md`](PLAN.md)
for the full build plan.

---

## The problem

Teams ship LLM features by pointing everything at the strongest available model.
It works, and the bill grows quietly until someone notices. Two things are missing:

1. **Nobody knows who is spending what.** Cost shows up as one provider invoice, not
   as spend per team or per feature.
2. **Most requests don't need the strongest model.** Extracting a date and writing a
   legal summary get billed at the same rate.

This gateway fixes both, and — importantly — measures whether the cheaper routing
actually held up on quality.

## How a request flows

```
POST /v1/chat
     |
     v
[1] classify      prompt -> complexity tier (1 simple / 2 moderate / 3 hard)
     |
[2] route         tier + feature overrides + context limits -> chosen model
     |
[3] estimate      count tokens, price with the model registry
     |
[4] reserve       atomically add the estimate to the day + month counters
     |            -> over budget? reject HERE, before spending anything
     v
[5] call provider (OpenAI / Anthropic / Ollama / mock)
     |
[6] settle        replace the estimate with the real cost, refund the difference
     |
[7] log           one usage row: tokens, latency, cost, and baseline cost
     v
   response  { text, model_used, tier, cost_usd, routing_reason, budget_status }
```

**The detail worth pausing on:** budgets are enforced *before* the provider call using
an estimate, but the true cost is only known *after* the response. So the gateway
reserves the estimate, then settles the difference once the real token counts arrive.
A failed call releases its reservation, so an error never silently eats budget.

## Design decisions

| Decision | Why |
|---|---|
| Estimate → reserve → settle | The only correct way to enforce a budget when cost is unknown until after the call. |
| Redis for counters, with an in-memory fallback | Budget checks sit in the hot path and must be atomic across workers. Unset `REDIS_URL` and it still runs. |
| Rule-based complexity classifier | Explainable and measurable. It returns the *reasons* for its tier, which a black box can't. |
| Shadow sampling against the strongest model | A savings number with no quality number just means "used the cheap model." |
| Baseline cost stored on every row | Savings are computed from data, not from a spreadsheet after the fact. |
| SQLite default, Postgres via Compose | Clone and run with no setup; production shape available with a config change. |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env

pytest                            # all tests run offline against the mock provider
```

No API keys are needed. The mock provider produces realistic token counts and
latency, so the test suite and the 1,000-prompt workload simulation both run free
and in seconds.

_Gateway, dashboard, and Docker Compose instructions land in Phases 5, 7, and 8._

## Project layout

```
app/          gateway, routing, budgets, providers, persistence
config/       models.yaml (registry) · routing.yaml (tiers) · budgets.yaml (limits)
dashboard/    Streamlit cost dashboard
eval/         hand-labeled prompts + classifier accuracy report
scripts/      workload simulation and reporting
tests/        one test file per module
```

## Known gaps

Called out deliberately rather than half-built:

- **No auth.** `team_id` is trusted from the request body. Real deployments would take
  it from a signed token.
- **No migrations.** Tables are created on startup; a production version needs Alembic.
- **Token estimates are approximate.** Estimation uses a single tokenizer across all
  providers. The savings report includes the estimate-vs-actual error so the size of
  that inaccuracy is visible instead of hidden.

## License

MIT — see [LICENSE](LICENSE).
