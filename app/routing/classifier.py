"""Request complexity classification.

Sorts a prompt into one of three tiers:

* **Tier 1** -- extraction, formatting, simple classification
* **Tier 2** -- summarization, rewriting, multi-step classification
* **Tier 3** -- reasoning, code generation, anything risky

This is a transparent scoring function, not a learned model, and that is a
deliberate choice: it returns the *reasons* for its decision, which a black box
cannot. Those reasons appear in the API response and the dashboard, so a routing
decision can be argued with rather than just accepted.

Being explainable does not make it correct. ``eval/classifier_eval.py`` scores it
against hand-labeled prompts and reports the cost of its mistakes, which is the
only reason to believe any of this works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.tokens import count_tokens

# --- signals -----------------------------------------------------------------
# Each group is a weight and a set of patterns. Positive weights push toward
# harder tiers, negative weights toward cheaper ones. Weights were calibrated
# against eval/labeled_prompts.jsonl -- see that report before changing any.

#: Pulling data out of text, with no judgment required.
EXTRACTION_VERBS = (
    r"\bextract\b", r"\bpull (?:out|the)\b", r"\blist\b", r"\bconvert\b",
    r"\bre-?format\b", r"\bformat these\b", r"\bformat this\b", r"\bparse\b",
    r"\btag\b", r"\bnormalize\b", r"\btranslate\b", r"\bcount how many\b",
    r"\bfind the\b", r"\bclassify\b",
    r"\bwhat is the (?:name|date|amount|email|total|due date)\b",
)

#: Genuine reasoning: causes, designs, judgments, trade-offs.
REASONING_VERBS = (
    r"\banaly[sz]", r"\bprove\b", r"\bdesign\b", r"\bevaluate\b", r"\bdebug\b",
    r"\brefactor\b", r"\bdiagnose\b", r"\bcritique\b", r"\brecommend\b",
    r"\bwhy\b", r"\broot cause\b", r"\btrade-?offs?\b",
    r"\bimplications?\b", r"\bstrategy\b", r"\barchitect", r"\bassess\b",
    r"\bvulnerabilit", r"\benforceable\b", r"\bdefensible\b",
    r"\bcredit risk\b", r"\bdata loss\b", r"\breconcil",
    r"\bwrite a \w+ function\b", r"\brollback plan\b",
)

#: Rewriting and restructuring: the classic tier-2 shape.
REWRITE_VERBS = (
    r"\bsummari[sz]", r"\brewrite\b", r"\bcondense\b", r"\bdraft\b",
    r"\bparaphrase\b", r"\bshorten\b", r"\bpolish\b", r"\breword\b",
    r"\bcompose\b", r"\bturn (?:this|these|it) .* into\b", r"\bcategori[sz]e\b",
    r"\bgroup (?:them|these)\b", r"\bprioriti[sz]e\b", r"\brank\b",
    r"\bwrite a (?:subject line|short|brief)\b",
)

#: Asking for justification -- pushes a mechanical task up into tier 2, and a
#: reasoning task up into tier 3 when both fire.
EXPLANATION_REQUESTS = (
    r"\bexplain\b", r"\band (?:tell|say) me why\b", r"\breason\b", r"\bjustif",
)

#: A requested machine-readable output usually means a mechanical task.
FORMAT_REQUESTS = (
    r"\bas json\b", r"\bjson\b", r"\bcsv\b", r"\bas a table\b",
    # Deliberately narrow: a bare "bullet points" usually describes the *input*
    # ("turn these bullet points into a paragraph"), not the requested output.
    r"\bone per line\b", r"\bas a bullet list\b", r"\bin bullets\b", r"\byaml\b",
)

CONSTRAINT_WORDS = (
    r"\bmust\b", r"\bensure\b", r"\brequired?\b", r"\bdo not\b", r"\bnever\b",
    r"\bcannot\b", r"\bwill not\b",
)

CODE_MARKERS = (
    r"```", r"\bdef \w+\(", r"\bfunction \w+\(", r"\bclass \w+[:\(]",
    r"\bSELECT\b.*\bFROM\b", r"\bALTER TABLE\b", r"\bstack ?trace\b",
    r"\btraceback\b", r"\bmigration\b", r"\bschema\b", r"\bindex\b",
    r"\bquery\b", r"\bfailed test", r"\bflaky\b", r"\brace condition\b",
    r"\bmemory leak\b",
)

COMPARISON_MARKERS = (
    r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bwhich is better\b",
    r"\bpros and cons\b", r"\bbuild or buy\b", r"\bagainst our requirements\b",
)

OPEN_ENDED_MARKERS = (
    r"\bhow should\b", r"\bwhat (?:is|'?s) the best\b", r"\bwhat should we\b",
    r"\bhow do we decide\b", r"\bhow to\b",
)

MULTI_STEP_MARKERS = (
    r"\bthen\b", r"\bafter that\b", r"\bstep \d\b", r"\bfinally\b",
    r"\bwithout (?:downtime|overwhelming)\b",
)

#: A prompt this long is doing more work regardless of its verbs.
LONG_PROMPT_TOKENS = 400

#: Score boundaries, calibrated against eval/labeled_prompts.jsonl.
TIER_1_MAX_SCORE = 0.5
TIER_2_MAX_SCORE = 2.0

#: Where an unrecognised prompt starts, and the most important number here.
#:
#: This lands in the tier-2 band, so a prompt matching no pattern at all routes
#: to a mid-tier model. Tier 1 has to be *earned* by positive evidence of a
#: mechanical task. Starting from zero instead -- which is what this did
#: originally -- meant every unfamiliar phrasing fell to the cheapest model, and
#: the holdout evaluation caught it: 36% accuracy with every single error an
#: under-route. Defaulting to the middle turns those failures into mild
#: over-spending instead of wrong answers.
BASELINE_SCORE = 1.0


@dataclass(frozen=True)
class Classification:
    """A tier, the score behind it, and why."""

    tier: int
    score: float
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """One line for the API response and the dashboard."""
        why = "; ".join(self.reasons) if self.reasons else "no strong signals"
        return f"tier {self.tier} (score {self.score:+.1f}): {why}"


def classify(prompt: str, risk_tags: list[str] | None = None) -> Classification:
    """Score `prompt` and return its tier with reasons.

    Risk tags dominate every text signal on purpose. If a caller has told us a
    request is legally or financially sensitive, no amount of "this looks like a
    simple extraction" should send it to the cheapest model.
    """
    risk_tags = risk_tags or []
    text = prompt.lower()
    tokens = count_tokens(prompt)

    score = BASELINE_SCORE
    reasons: list[str] = []
    features: dict[str, float] = {"prompt_tokens": float(tokens)}

    def add(weight: float, reason: str, feature: str, value: float = 1.0) -> None:
        nonlocal score
        score += weight
        reasons.append(reason)
        features[feature] = value

    if risk_tags:
        add(2.5, f"risk tags {sorted(risk_tags)}", "risk_tags", float(len(risk_tags)))

    # A single strong reasoning verb clears the tier-3 threshold on its own.
    # Calibration showed the alternative -- requiring a second signal -- sent
    # "Design a retry strategy" and "Root cause this spike" to tier 2, which is
    # the expensive kind of mistake.
    has_reasoning = _matches_any(text, REASONING_VERBS)
    if has_reasoning:
        add(2.1, "reasoning verb", "reasoning_verb")

    has_code = _matches_any(text, CODE_MARKERS)
    if has_code:
        add(1.0, "code or query content", "code")

    # "What should we do about X" is a judgment call by construction.
    has_open_ended = _matches_any(text, OPEN_ENDED_MARKERS)
    if has_open_ended:
        add(2.1, "open-ended question", "open_ended")

    if _matches_any(text, COMPARISON_MARKERS):
        add(0.8, "asks for a comparison", "comparison")

    if _count_matches(text, MULTI_STEP_MARKERS) >= 1 and (has_reasoning or has_code):
        add(0.8, "multi-step instructions", "multi_step")

    has_rewrite = _matches_any(text, REWRITE_VERBS)
    if has_rewrite:
        add(0.7, "rewrite or summarize verb", "rewrite_verb")

    has_explanation = _matches_any(text, EXPLANATION_REQUESTS)
    if has_explanation:
        add(0.7, "asks for an explanation", "explanation_request")

    constraints = _count_matches(text, CONSTRAINT_WORDS)
    if constraints >= 2:
        add(0.7, f"{constraints} hard constraints", "constraints", float(constraints))

    if tokens > LONG_PROMPT_TOKENS:
        add(0.8, f"long prompt ({tokens} tokens)", "long_prompt", float(tokens))

    # Only counts when it is the *dominant* instruction. "Summarize this thread
    # and list the open questions" is a summarization task that happens to
    # contain the word "list"; the discount would otherwise cancel the rewrite
    # signal and send it to the cheapest model.
    if _matches_any(text, EXTRACTION_VERBS) and not (has_rewrite or has_reasoning):
        add(-1.0, "extraction verb", "extraction_verb")

    if _matches_any(text, FORMAT_REQUESTS):
        add(-0.5, "machine-readable output requested", "format_request")

    # There is deliberately no "short prompt" discount. It was here originally,
    # and the holdout evaluation showed it was the single largest source of
    # under-routing: instructions are terse whatever their difficulty, so
    # "Recap the decisions in this thread" and "Extract the invoice number" are
    # the same length. Brevity says nothing about how hard a task is.

    return Classification(
        tier=_tier_for_score(score),
        score=round(score, 2),
        reasons=reasons,
        features=features,
    )


def _tier_for_score(score: float) -> int:
    if score <= TIER_1_MAX_SCORE:
        return 1
    if score <= TIER_2_MAX_SCORE:
        return 2
    return 3


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
