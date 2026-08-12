# LLM Spend Control Center — One-Day Build Plan

A routing and budgeting layer that sits in front of LLM calls. It tracks spend per
team and feature, picks the cheapest model that can still do the job, and blocks
requests when a budget runs out.

**Headline metric we are building toward:**
> Cut simulated LLM spend by X% while holding a Y% quality pass rate against the
> strongest model.

---

## Ground rules for the whole build

**Code style**
- Python 3.11, type hints everywhere, no clever tricks.
- Every file starts with a short docstring: what it does and who calls it.
- One job per file. If a file passes ~150 lines, split it.
- Cost math and routing rules are **pure functions** — easy to test, no I/O inside.
- Config lives in YAML, never hardcoded in Python.
- No secrets in code. Everything through `.env`.

**GitHub-ready from the first commit**
- `.gitignore`, `LICENSE`, `README.md`, `.env.example`, and CI exist in Phase 0.
- Every phase ends with: tests green → commit → push. Nine commits, nine pushes.
- Commit messages follow `type: summary` (`feat:`, `test:`, `docs:`, `chore:`).
- CI runs `pytest` on every push, so a broken phase is visible on GitHub.

**Keeping it free and fast**
- A `MockProvider` returns fake responses with realistic token counts. All tests and
  the 1,000-prompt simulation run on it, so the demo costs $0 and finishes in seconds.
- Real providers (OpenAI, Anthropic, Ollama) are wired and work, but are opt-in via env.
- Responses are cached by `hash(model + prompt)` so re-runs never pay twice.

**Infra that runs with zero setup**
- `DATABASE_URL` defaults to SQLite so `uvicorn app.main:app` just works.
- Docker Compose swaps in Postgres + Redis with no code change.
- Same for counters: Redis if `REDIS_URL` is set, in-memory dict if not.

---

## Final file tree

```
llm-spend-control/
├─ .github/workflows/ci.yml     # pytest on push
├─ .env.example
├─ .gitignore
├─ LICENSE
├─ README.md
├─ PLAN.md
├─ docker-compose.yml           # gateway + postgres + redis + dashboard
├─ Dockerfile
├─ requirements.txt
├─ pytest.ini
│
├─ config/
│  ├─ models.yaml               # model registry: cost, tier, latency, context
│  ├─ routing.yaml              # tier -> model, plus per-feature overrides
│  └─ budgets.yaml              # daily + monthly limits per team and feature
│
├─ app/
│  ├─ settings.py               # env config in one place
│  ├─ schemas.py                # unified request / response shapes
│  ├─ registry.py               # loads models.yaml, does cost math
│  ├─ main.py                   # FastAPI app, POST /v1/chat
│  ├─ reporting.py             # spend, savings, and routing-quality queries
│  ├─ providers/
│  │  ├─ base.py                # Provider interface + Completion result
│  │  ├─ mock.py                # free, deterministic, used by tests + sim
│  │  ├─ openai_provider.py
│  │  ├─ anthropic_provider.py
│  │  └─ ollama_provider.py
│  ├─ routing/
│  │  ├─ classifier.py          # prompt -> complexity tier
│  │  └─ router.py              # tier + overrides -> chosen model
│  ├─ budget/
│  │  ├─ counters.py            # Redis or in-memory spend counters
│  │  └─ enforcer.py            # estimate -> reserve -> settle
│  ├─ quality/
│  │  ├─ sampler.py             # decides which requests get shadow-checked
│  │  └─ judge.py               # compares cheap answer vs strong answer
│  └─ db/
│     ├─ models.py              # SQLAlchemy tables
│     └─ repository.py          # every DB read/write lives here
│
├─ dashboard/app.py             # Streamlit cost dashboard
├─ eval/
│  ├─ labeled_prompts.jsonl     # ~120 hand-labeled tier examples
│  └─ classifier_eval.py        # accuracy + cost of misroutes
├─ scripts/simulate_workload.py # 1,000 prompts -> savings report
├─ reports/                     # generated output (gitignored)
├─ docs/CASE_STUDY.md
└─ tests/                       # one test file per module
```

---

## Phase 0 — Repo skeleton (~30 min)

**Goal:** an empty but professional repo that already passes CI.

Build:
- `.gitignore` (venv, `__pycache__`, `.env`, `*.db`, `reports/*`)
- `requirements.txt` — fastapi, uvicorn, pydantic, pydantic-settings, sqlalchemy,
  psycopg2-binary, redis, pyyaml, httpx, streamlit, pandas, pytest, respx
- `.env.example` — every env var with a safe default and a comment
- `README.md` — problem, architecture diagram, quickstart (placeholders for numbers)
- `LICENSE` (MIT), `pytest.ini`
- `.github/workflows/ci.yml` — install deps, run pytest on push and PR
- `app/settings.py` — one `Settings` class reading env
- Empty packages with `__init__.py`, plus `tests/test_settings.py` so CI has something to run

**Done when:** `pytest` passes, `git push` succeeds, CI is green on GitHub.
**Commit:** `chore: project skeleton, CI, and settings`

---

## Phase 1 — Model registry and provider adapters (~60 min)

**Goal:** one request shape and one response shape, no matter which provider ran it.

Build:
- `config/models.yaml` — for each model: provider, tier (1/2/3), input cost per 1M
  tokens, output cost per 1M, typical latency, max context, supports vision / tools.
- `app/registry.py` — load and validate the YAML; `cost_of(model, in_tokens, out_tokens)`;
  `models_for_tier(tier)`. Pure functions, no network.
- `app/schemas.py` — `ChatRequest` (messages, team_id, feature, priority,
  optional model preference, risk tags) and `ChatResponse` (text, model used, tier,
  tokens, latency_ms, cost_usd, routing reason, budget status).
- `app/providers/base.py` — `Provider` protocol returning a `Completion`
  (text, input_tokens, output_tokens, latency_ms, raw provider metadata).
- `app/providers/mock.py` — deterministic fake output, token counts derived from
  prompt length, fake latency scaled by tier.
- `openai_provider.py`, `anthropic_provider.py`, `ollama_provider.py` — thin `httpx`
  calls that map provider fields into `Completion`. Nothing else.

Tests: cost math (including a hand-checked worked example), YAML validation errors,
mock provider determinism, one mocked HTTP test per real provider using `respx`.

**Done when:** every provider returns the identical `Completion` shape.
**Commit:** `feat: model registry and unified provider adapters`

---

## Phase 2 — Usage log in the database (~45 min)

**Goal:** every request is a queryable row. This table is the whole project's evidence.

Build:
- `app/db/models.py` — three tables:
  - `usage_events`: timestamp, team_id, feature, priority, model, tier, chosen_reason,
    input/output tokens, latency_ms, status, cost_usd, **baseline_cost_usd**
    (what the strongest model would have cost — this is how savings get measured later)
  - `budget_ledger`: reservations and settlements, so estimate-vs-actual is auditable
  - `routing_misses`: prompt, chosen model, better model, judge reason, score
- `app/db/repository.py` — `log_usage()`, `record_miss()`, plus the read queries the
  dashboard needs. **All SQL lives here** — nothing else in the codebase touches the DB.
- Table creation on startup (no Alembic; it's overkill for one day).

Tests: write then read back a usage event; `spend_today()` and `spend_this_month()`
return the right totals against seeded rows.

**Done when:** rows land in SQLite locally and Postgres under Compose, unchanged code.
**Commit:** `feat: durable usage log and repository layer`

---

## Phase 3 — Budgets: warn at 80%, block at 100% (~60 min)

**Goal:** the systems-thinking centrepiece. Check the budget *before* the call using an
estimate, then correct it *after* with the real cost.

The flow, in plain words:
1. **Estimate** — token-count the prompt, guess output length, price it with the registry.
2. **Reserve** — atomically add the estimate to today's and this month's counters. If it
   would cross the limit, reject now, before any money is spent.
3. **Call** the provider.
4. **Settle** — replace the estimate with the real cost (`reserved - actual` is refunded).
   A crashed call refunds the full reservation, so a failure never eats budget.

Build:
- `config/budgets.yaml` — daily and monthly caps per team, optional per-feature caps.
- `app/budget/counters.py` — `SpendCounters` with `reserve()`, `settle()`, `release()`,
  `current()`. Redis backend (atomic `INCRBYFLOAT` + day/month keys with TTL) and an
  in-memory backend with the same interface.
- `app/budget/enforcer.py` — the decision logic as a pure function returning
  `ALLOW` / `WARN` / `BLOCK` plus a human-readable reason:
  - under 80% → `ALLOW`
  - 80–100% → `WARN` (request proceeds, warning attached to the response)
  - over 100% → `BLOCK` for `low`/`normal` priority; `high` priority passes with an
    override flag that gets recorded in the ledger

Tests: reserve → settle leaves the counter at the actual cost; reserve → release leaves
it at zero; the 79/80/100/101 percent boundaries; a high-priority override at 105%;
concurrent reservations don't oversell the budget.

**Done when:** blocked requests return a clear error naming the limit, the current spend,
and how to override — never a silent failure.
**Commit:** `feat: pre-call budget enforcement with post-call reconciliation`

---

## Phase 4 — Complexity routing (~60 min)

**Goal:** send easy work to cheap models, hard work to strong models, and be able to
*prove* the classifier is any good.

Three tiers:
- **Tier 1** — extraction, formatting, simple classification
- **Tier 2** — summarization, rewriting, multi-step classification
- **Tier 3** — reasoning, code generation, anything risky

Build:
- `app/routing/classifier.py` — a transparent scoring function over readable features:
  prompt length, instruction verbs (`explain`/`analyze`/`prove` push up,
  `extract`/`list`/`format` push down), required output format, context size,
  number of constraints, and risk tags. Returns `(tier, score, reasons[])` — the
  reasons string is what makes the dashboard and the interview story convincing.
- `config/routing.yaml` — tier → preferred model + fallbacks, and **override rules**:
  named features that always get Tier 3 because correctness beats cost.
- `app/routing/router.py` — resolves the final model from: explicit user preference →
  feature override → classifier tier → context-length and capability filtering →
  provider availability. Returns the model plus the reason it won.
- `eval/labeled_prompts.jsonl` — ~120 prompts hand-labeled with their true tier.
- `eval/classifier_eval.py` — prints accuracy, a confusion matrix, and the
  **asymmetric cost of mistakes**: routing a Tier-3 request down to Tier 1 is a quality
  failure, while routing Tier 1 up to Tier 3 only wastes money. These are not equally bad
  and the report says so.

Tests: each tier's obvious cases, override rules beat the classifier, a request too long
for the cheap model's context gets upgraded, unavailable provider falls back cleanly.

**Done when:** `python -m eval.classifier_eval` prints accuracy and misroute costs.
**Commit:** `feat: complexity classifier and tier-based model router`

---

## Phase 5 — Wire the gateway end to end (~45 min)

**Goal:** one endpoint that does all of the above in the right order.

`POST /v1/chat` sequence: classify → route → estimate → reserve → call provider →
settle → log usage → return. Every step's outcome shows up in the response, so the
demo is self-explaining.

Build:
- `app/main.py` — the endpoint, dependency wiring, and clean error mapping:
  `402` budget exceeded, `503` provider down, `422` bad request. Every error body says
  what happened and what to do.
- `GET /v1/budgets/{team_id}` — current spend, limits, percent used.
- `GET /health` — DB, Redis, and provider reachability.
- Response includes `routing_reason` and `budget_status` so a human can read the decision.

Tests: happy path with the mock provider; a blocked request writes a `blocked` usage row
at zero cost plus a ledger entry; a provider failure refunds the reservation; the response
schema is stable.

> **Changed during Phase 2.** This originally said a blocked request writes *no* usage row.
> It now writes one with `status = "blocked"` and `cost_usd = 0`. Without rows for failed
> attempts there is no way to report a block rate or an error rate by provider, and an audit
> log that only records successes is not much of an audit log. Spend queries filter on
> `status = "ok"`, so blocked and errored rows never inflate a total.

**Done when:** `curl` a request and see the model choice, cost, and budget state come back.
**Commit:** `feat: unified chat gateway endpoint`

---

## Phase 6 — Quality checks and escalation (~45 min)

**Goal:** make the savings claim honest. Savings without a quality number is just
"I used the cheap model."

Build:
- `app/quality/sampler.py` — pick ~10% of cheap-model requests for shadow checking
  (rate from config, deterministic hash so runs are reproducible).
- `app/quality/judge.py` — run the same prompt on the strongest model, ask a judge to
  score the cheap answer, and store a pass/fail with a reason. Runs in a FastAPI
  background task so the user never waits for it.
- On failure, write a `routing_misses` row: prompt, chosen model, better model, reason.
- **Auto-escalation, before responding:** if the request is `high` priority or carries a
  risk tag, or the cheap model's output looks degenerate (empty, truncated, refusal-like),
  rerun on the stronger model and return that instead. Log both attempts and both costs —
  escalation is not free and the report should admit it.

Tests: sampling rate is deterministic; a failed judge creates exactly one miss row;
escalation triggers on high priority and logs two usage events; judge failures never
break the user's request.

**Done when:** the verifier pass rate is a real measured number, not an assumption.
**Commit:** `feat: shadow quality sampling and auto-escalation`

---

## Phase 7 — Cost dashboard (~45 min)

**Goal:** the screenshots that go in the README.

Build `app/reporting.py` (all queries, one function each) and `dashboard/app.py`
(Streamlit, four tabs, thin — presentation only, zero logic):

1. **Spend** — daily cost by team and feature, month-to-date, end-of-month projection,
   cost split by model, top 10 most expensive prompts.
2. **Savings** — actual routed spend vs. the "everything to the strongest model"
   baseline, as a running chart plus one big headline percentage. This is the money shot.
3. **Routing quality** — tier distribution, escalation rate, verifier pass rate,
   routing-miss list with reasons.
4. **Ops** — latency by model (p50/p95), error rate by provider, budget gauges per team.

**Done when:** every number on screen traces back to a query in `reporting.py`.
**Commit:** `feat: streamlit cost and routing dashboard`

---

## Phase 8 — Simulation, Docker, case study (~60 min)

**Goal:** the artifacts a reviewer actually looks at.

Build:
- `scripts/simulate_workload.py` — 1,000 mixed prompts (a spread of tiers, teams,
  features, and priorities; sourced from a public dataset where possible rather than
  invented, so the distribution isn't rigged in our favour). Runs on the mock provider,
  writes `reports/savings_report.md` + a CSV: total routed cost, baseline cost,
  savings %, tier distribution, escalation rate, verifier pass rate, p95 latency.
- `docker-compose.yml` + `Dockerfile` — gateway, Postgres, Redis, dashboard. One command up.
- `docs/CASE_STUDY.md` — the real numbers from the simulation, the routing design, the
  budget enforcement flow, what the classifier gets wrong, and what you'd do differently
  at production scale.
- Finish `README.md`: headline result, architecture diagram, quickstart, dashboard
  screenshots, and the design decisions worth defending.

**Done when:** a stranger can clone, run `docker compose up`, and see the numbers.
**Commit:** `docs: workload simulation results and case study`

---

## Schedule

| Phase | Work | Time |
|---|---|---|
| 0 | Repo skeleton + CI | 0:30 |
| 1 | Registry + providers | 1:00 |
| 2 | Usage log | 0:45 |
| 3 | Budgets | 1:00 |
| 4 | Routing | 1:00 |
| 5 | Gateway endpoint | 0:45 |
| 6 | Quality + escalation | 0:45 |
| 7 | Dashboard | 0:45 |
| 8 | Simulation + case study | 1:00 |
| | **Total** | **~8:00** |

## Deliberately cut

- **Celery / RQ** — FastAPI background tasks cover the async verification. Celery is two
  hours of wiring for no extra interview signal.
- **Alembic migrations** — `create_all` is honest for a portfolio project.
- **Auth / multi-tenancy** — `team_id` comes from the request body. Noted in the README
  as a known production gap rather than half-built.
- **A trained sklearn classifier** — the rule-based scorer is explainable and measurable.
  Swapping in logistic regression later is a stretch goal, not a day-one requirement.

## Where this build could go wrong

- **Fake savings.** Cheap models are cheaper by definition, so a savings number alone
  proves nothing. Phase 6's measured pass rate is what makes it defensible — do not skip it.
- **Untested classifier.** Without `eval/labeled_prompts.jsonl` the routing story is
  unfalsifiable. 120 labels is an hour well spent.
- **Estimate drift.** If estimated cost is far off actual, budget enforcement is theatre.
  The report should include estimate-vs-actual error, and own it if it's bad.
