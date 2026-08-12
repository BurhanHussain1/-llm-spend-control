"""Streamlit cost dashboard.

Run it:

    streamlit run dashboard/cost_dashboard.py

Presentation only. Every figure on screen comes from ``app/reporting.py`` or
``app/db/repository.py``, so nothing can be computed one way here and another way
in the case study.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import reporting
from app.db.engine import create_tables, get_session_factory
from app.db.repository import Repository, start_of_utc_month
from app.dependencies import get_budget_policies

st.set_page_config(page_title="LLM Spend Control Center", layout="wide")


@st.cache_resource
def session_factory():
    create_tables()
    return get_session_factory()


def repository() -> Repository:
    return Repository(session_factory()())


def money(value: float) -> str:
    """Format a dollar amount, keeping small figures legible.

    Individual requests can cost a hundredth of a cent, and rounding those to two
    decimal places would display most of this system's traffic as $0.00.
    """
    if value and abs(value) < 0.01:
        return f"${value:.6f}"
    return f"${value:,.2f}"


repo = repository()
scope = st.sidebar.radio("Period", ["Month to date", "All time"], index=0)
since = start_of_utc_month() if scope == "Month to date" else None

st.title("LLM Spend Control Center")

savings = reporting.savings_summary(repo, since)
if not savings.requests:
    st.info(
        "No usage recorded yet. Send a request to `POST /v1/chat`, or run "
        "`python -m scripts.simulate_workload` to generate a workload."
    )
    st.stop()

spend_tab, savings_tab, quality_tab, ops_tab = st.tabs(
    ["Spend", "Savings", "Routing quality", "Ops"]
)

# --- Spend -------------------------------------------------------------------

with spend_tab:
    left, middle, right = st.columns(3)
    left.metric("Requests", f"{savings.requests:,}")
    middle.metric("Routed spend", money(savings.routed_cost_usd))
    right.metric("Provider calls", f"{savings.provider_calls:,}")

    st.subheader("By team")
    st.dataframe(
        pd.DataFrame(
            repo.spend_by("team", since), columns=["Team", "Requests", "Cost (USD)"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("By feature")
    st.dataframe(
        pd.DataFrame(
            repo.spend_by("feature", since),
            columns=["Feature", "Requests", "Cost (USD)"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("By model")
    st.dataframe(
        pd.DataFrame(
            repo.spend_by("model", since), columns=["Model", "Requests", "Cost (USD)"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Month-end projection")
    projections = [
        {"Team": team, **reporting.month_end_projection(repo, team)}
        for team in get_budget_policies().known_teams()
    ]
    st.dataframe(pd.DataFrame(projections), use_container_width=True, hide_index=True)
    st.caption(
        "Straight-line extrapolation from the current daily rate. Early in a "
        "month it swings wildly, which is a property of the arithmetic and not a "
        "signal."
    )

    st.subheader("Most expensive requests")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "When": event.created_at,
                    "Team": event.team_id,
                    "Feature": event.feature,
                    "Model": event.model,
                    "Cost (USD)": event.cost_usd,
                    "Prompt": event.prompt_preview[:110],
                }
                for event in repo.most_expensive(10)
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

# --- Savings -----------------------------------------------------------------

with savings_tab:
    st.subheader("Against sending everything to the strongest model")

    a, b, c = st.columns(3)
    a.metric("Baseline cost", money(savings.baseline_cost_usd))
    b.metric("Total spend", money(savings.total_spend_usd))
    c.metric(
        "Net saving",
        f"{savings.net_savings_percent:.1f}%",
        delta=money(savings.net_savings_usd),
    )

    st.markdown(
        f"""
| | Amount | Note |
|---|---|---|
| Baseline (all requests on the strongest model) | {money(savings.baseline_cost_usd)} | the counterfactual |
| Routed spend | {money(savings.routed_cost_usd)} | what the router chose |
| Escalation | {money(savings.escalation_cost_usd)} | reruns after an unusable answer |
| Verification | {money(savings.verification_cost_usd)} | shadow checks proving the quality claim |
| **Total spend** | **{money(savings.total_spend_usd)}** | routed + escalation + verification |
| Gross saving | {money(savings.gross_savings_usd)} ({savings.gross_savings_percent:.1f}%) | ignores operating cost |
| **Net saving** | **{money(savings.net_savings_usd)} ({savings.net_savings_percent:.1f}%)** | **the honest number** |
"""
    )
    st.caption(
        f"Escalation and verification together add "
        f"{savings.overhead_percent_of_routed:.1f}% on top of routed spend. "
        "Quote the net figure: a system that saves 70% and spends 40% checking "
        "itself has not saved 70%."
    )

    daily = repo.daily_spend(since)
    if daily:
        st.subheader("Daily: actual vs baseline")
        frame = pd.DataFrame(daily, columns=["Date", "Actual", "Baseline"]).set_index(
            "Date"
        )
        st.line_chart(frame)

    st.subheader("Estimate accuracy")
    error = reporting.estimate_error(repo, since)
    left, middle, right = st.columns(3)
    left.metric("Median absolute error", f"{error['median_absolute_percent']:.1f}%")
    middle.metric("Median signed error", f"{error['median_signed_percent']:+.1f}%")
    right.metric("Mean absolute error", f"{error['mean_absolute_percent']:.1f}%")
    st.caption(
        "Budgets are enforced against the pre-call estimate, so this is the "
        "honesty check on that enforcement. A positive median means estimates run "
        "high and budget is briefly over-reserved, then handed back on settle. "
        "**Read the median, not the mean:** percentage error divides by actual "
        "cost, so one request that produced almost no output can push the mean "
        "into the thousands of percent while the typical request is within 10%. "
        f"Sample: {error['samples']:,} requests."
    )

# --- Routing quality ---------------------------------------------------------

with quality_tab:
    verification = reporting.verification_summary(repo, since)

    a, b, c = st.columns(3)
    a.metric(
        "Verifier pass rate",
        f"{verification.pass_rate_percent:.1f}%" if verification.graded else "n/a",
    )
    b.metric("Graded checks", f"{verification.graded:,}")
    c.metric("Escalation rate", f"{reporting.escalation_rate(repo, since):.1f}%")

    if verification.is_mechanical_only or not verification.graded:
        st.warning(verification.caveat)
    else:
        st.caption(verification.caveat)

    if verification.not_run:
        st.caption(
            f"{verification.not_run:,} checks could not be run (reference model "
            "unavailable, or the verification itself was refused by a budget). "
            "They are excluded from the pass rate rather than counted as failures."
        )

    st.subheader("Tier distribution")
    tiers = repo.tier_distribution(since)
    if tiers:
        st.bar_chart(
            pd.DataFrame(
                {"Requests": list(tiers.values())},
                index=[f"Tier {tier}" for tier in tiers],
            )
        )

    st.subheader("Routing misses")
    misses = repo.routing_misses(20)
    if misses:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When": miss.created_at,
                        "Chose": miss.chosen_model,
                        "Should have used": miss.better_model,
                        "Score": miss.score,
                        "Why": miss.reason[:120],
                        "Prompt": miss.prompt[:100],
                    }
                    for miss in misses
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No routing misses recorded.")

# --- Ops ---------------------------------------------------------------------

with ops_tab:
    st.subheader("Latency by model")
    latency = reporting.latency_percentiles(repo, since)
    if latency:
        st.dataframe(pd.DataFrame(latency), use_container_width=True, hide_index=True)

    st.subheader("Outcomes by provider")
    st.dataframe(
        pd.DataFrame(reporting.error_rates(repo, since)),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "`blocked` counts requests refused by a budget before any provider call, "
        "so they are not provider failures."
    )

    st.subheader("Budget usage")
    st.caption(
        "Read from the usage log rather than the live counters, so this page needs "
        "no connection to Redis."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Team": team,
                    "Today (USD)": repo.spend_today(team),
                    "Daily limit": get_budget_policies().for_team(team).daily_limit_usd,
                    "Month (USD)": repo.spend_this_month(team),
                    "Monthly limit": get_budget_policies()
                    .for_team(team)
                    .monthly_limit_usd,
                }
                for team in get_budget_policies().known_teams()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
