# Case study: routing and budgeting 1,000 LLM requests

## Result

**Cut simulated LLM spend by 15.2% net (19.9% gross) across 1,000 requests, at a
100% verifier pass rate on 14 mechanically graded samples.**

Both halves of that sentence need qualifying, and the qualifications are the
interesting part.

| | |
|---|---|
| Baseline — every request on the strongest model | $6.0981 |
| Routed spend | $4.8851 |
| Escalation (reruns after an unusable answer) | $0.1617 |
| Verification (shadow checks that produced the quality number) | $0.1258 |
| **Total spend** | **$5.1726** |
| Gross saving | $1.2131 (19.9%) |
| **Net saving** | **$0.9255 (15.2%)** |

Reproduce it with `python -m scripts.simulate_workload`; the full report lands in
`reports/savings_report.md`.

## Why it is 15% and not 68%

A single tier-1 request in this system costs $0.000363 against a baseline of
$0.001815 — an 80% saving. Scale that to a realistic mix and it collapses to 20%.
The reason is visible in one table:

| Tier | Requests | Share of requests | Share of routed spend |
|---|---|---|---|
| 1 | 186 | 18.6% | 1.4% |
| 2 | 348 | 34.8% | 12.8% |
| 3 | 498 | 46.6% | **91.7%** |

**The savings ceiling is set by the traffic mix, not by how clever the router is.**
Half the traffic needs a strong model, and that half spends 92% of the money. A
router can only ever compete for the other 8%. Getting every tier-1 decision
perfect — which this system does, on this corpus — moves the total by a few
percent, because tier-1 requests are cheap by definition.

This is the number to be honest about. It would have been easy to build a corpus
of extraction prompts, report 70%, and never mention the mix.

## What the safety rails cost

Tier 3 took 46.6% of requests, but only ~33% of the corpus is labelled tier 3. The
gap is the deliberate override rules:

- `refund_decision` and `incident_postmortem` are pinned to tier 3 whatever the
  prompt looks like — 25% of the simulated traffic, forced upward.
- `customer_summary` carries a tier-2 floor.
- Any risk tag escalates to tier 3.

Those rules are why a five-token question about a refund goes to the strongest
model. **They are also where most of the potential saving went.** That trade is
the design, not a defect: the cost of one wrong refund answer exceeds the cost of
routing every refund question to Opus for a month. But it should be stated as a
trade, with a number attached, rather than hidden.

## What verification cost

Verification added 5.9% on top of routed spend, and getting there took a
correction. Verifying one tier-1 request costs about **25×** what the request
itself cost — the reference model is priced ~5× higher *and* writes a longer
answer to the same prompt. At the 10% sample rate originally planned, that works
out to ~2.5× the routed cost of every tier-1 request, which would have cut
reported savings from ~80% to ~31% on tier-1 traffic:

> **The measurement would have consumed most of what it was measuring.**

Sampling is 2% and the report shows verification as its own line. A savings figure
that hides the cost of its own evidence is not a savings figure.

## What the quality number is worth

100% of 14 graded samples passed. Read that as a smoke test, not a quality
measurement:

- The judge was **mechanical** — with no provider credentials it detects empty and
  truncated answers and cannot assess correctness. Every verdict is stored with
  the judge's name so the number can never be quoted without its caveat.
- 14 samples is a small denominator. It is 2% of the cheap-model traffic by design.
- A real quality figure needs an API key and the LLM judge, which grades the cheap
  answer against the strong one on a rubric.

Two metric definitions were fixed to stop this number flattering itself:

1. **Skipped checks are `NULL`, not `False`.** If the reference model is down or a
   budget refuses the verification, that says nothing about the answer. Counting
   those as failures would understate the pass rate for unrelated reasons; the
   pass rate uses `WHERE passed IS NOT NULL`.
2. **The length floor was wrong.** The mechanical judge originally failed any
   answer under 25% of the reference's length. Mock cheap answers are ~19% of the
   reference by construction, so the pass rate would have been ~0% — measuring
   verbosity, not quality. A four-word answer to an extraction task beats a
   four-sentence one.

## The budget design

Budgets are enforced **before** the provider call and reconciled **after**,
because the true cost is only known once the tokens come back:

```
estimate → reserve → call provider → settle (or release)
```

The reservation **increments the counter first, then inspects the result**.
Read-then-write would let two concurrent requests each see room for one; there is
a test that races ten requests at $0.40 against a $1.00 budget and asserts exactly
two get through. A blocked request rolls its own increment back, so a flood of
rejections cannot exhaust a budget that had room, and a provider failure releases
the whole reservation — an outage never eats a team's budget.

### The estimator is biased low, and that is a real weakness

Measured on the run, across 1,032 calls: **median absolute error 62.8%, median
signed error −62.8%.** The estimates are not noisy around the truth — they are
systematically *under* by about two thirds.

That matters, because it is the estimate the budget is held against. The
consequence is bounded but real: a team can overshoot its limit on the request
that crosses the line by roughly 2.7×, and only the post-call settle brings the
counter back to truth. Enforcement is directionally correct and it is not the
"theatre" it would be if the error were unbounded — but "we enforce budgets before
the call" deserves the asterisk.

The cause is the output-length guess. The estimator assumes output tokens are
`max(64, input × 0.5)`, while tier-2 and tier-3 answers in this workload run to
several hundred tokens. Input tokens are counted accurately; output is where it
goes wrong, and output is the more expensive half of the bill.

The fix is available and unbuilt: the usage log already records actual output
tokens per model, so the estimator should use each model's observed median instead
of a fixed ratio. That is the highest-value single improvement left in this
project, and I would rather report the 62.8% than quietly widen the reservation to
hide it.

One metric note: the *mean* absolute error tracks the median here (61.8%), but on
smaller samples it does not — an early run showed mean 1495% against median 8.5%,
because percentage error divides by actual cost and one near-empty response
dominates. The dashboard leads with the median for that reason.

## The classifier is the weakest part

| Set | Accuracy |
|---|---|
| Calibration (127 prompts, weights tuned on it) | 98.4% |
| **Holdout (33 prompts, never tuned against)** | **48.5%** |

I first scored 85%, tuned to 100%, then recognised 100% as evidence of
overfitting rather than skill. A holdout written afterwards in different phrasing
scored **36.4%, with 63.6% of prompts under-routed.**

That exposed a structural defect: an unrecognised prompt defaulted to **tier 1**,
the cheapest model — backwards for a system where under-routing is the expensive
mistake. Unknown prompts now start in the tier-2 band and tier 1 must be earned by
positive evidence of a mechanical task. The "short prompt" discount was deleted
outright: instructions are terse whatever their difficulty. Holdout moved to
48.5%, and under-routing fell to 21.2% with the difference becoming harmless
over-routing.

I stopped tuning there. Further changes would fit a holdout I had already read.

**So the honest conclusion is that a ~50% classifier is not what makes this system
safe.** The feature overrides, the risk-tag floors, and the shadow verification
are. The classifier is a cost optimisation sitting behind three safety nets — and
that is only visible because the holdout exists.

## What I would do differently at production scale

- **Fix the cost estimator first.** Replace the fixed output-length ratio with each
  model's observed median output tokens from the usage log. Estimates are 62.8%
  low today, which is the one measured weakness in the budget-enforcement claim.
- **Replace the classifier.** A rule-based scorer at 48.5% generalization is a
  starting point. The labelled corpus is the asset; a small trained model over the
  same features, retrained on production traffic with the routing-miss table as
  labels, is the obvious next step.
- **Segment the savings claim by feature.** A blended 15% hides the fact that
  extraction-heavy features save 80% and reasoning-heavy features save nothing.
  Per-feature reporting would drive better decisions than one headline.
- **Store money as `Numeric`, not `Float`.** Summing floats produced
  `$0.01 + $0.05 = 0.060000000000000005`; sums are rounded at the query boundary,
  which is a patch on the column type.
- **Add Alembic.** Tables are created on startup, which is fine for a demo and
  wrong for a service that has to change its schema without downtime.
- **Take `team_id` from a signed token.** It is trusted from the request body
  today, so any caller can spend another team's budget.
- **Move verification off the request path entirely.** `BackgroundTasks` dies with
  the process; a durable queue would survive a deploy mid-check.

## Limitations of this simulation

1. **The prompt mix is ours.** Prompts come from this repo's hand-labelled corpus,
   not a public dataset, so the tier distribution — and therefore the savings
   percentage — reflects a mix we chose.
2. **No provider was called.** `mock-cheap`/`mid`/`strong` are priced identically
   to the Haiku/Sonnet/Opus tiers, so cost ratios are realistic, but latency is
   synthetic and answer quality is not real.
3. **Budgets ran against one logical day.** Log timestamps are spread over 14 days
   for charting; enforcement during the run used a single "now", so the zero block
   count is not a multi-day block rate. Blocking behaviour is covered by tests.
