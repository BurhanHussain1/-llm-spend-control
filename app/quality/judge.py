"""Scoring a cheap model's answer against a stronger model's.

Two judges, because the honest answer depends on what is actually running:

* :class:`LLMJudge` -- asks the strongest configured model to score the cheap
  answer against its own. This is the real thing, and it needs a provider key.
* :class:`MechanicalJudge` -- offline fallback for runs with no credentials. It
  checks only what can be checked without understanding: emptiness, truncation,
  and gross length mismatch against the reference.

**This distinction has to be reported, not buried.** A pass rate produced by the
mechanical judge is a smoke test, not a quality measurement -- it cannot tell a
wrong answer from a right one. Every verdict therefore carries the name of the
judge that produced it, and the savings report prints which judge ran. A "94%
verifier pass rate" means nothing if a regex produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.providers.base import Provider, ProviderError
from app.registry import Model
from app.schemas import Message

#: Below this score, the cheap model's answer is treated as a routing miss.
DEFAULT_PASS_THRESHOLD = 0.7

#: How much shorter than the reference an answer may be before it looks truncated.
#:
#: Deliberately very low. A shorter answer is frequently the *better* one -- an
#: extraction task answered in four words against a reference that padded to four
#: sentences is a win, not a miss. Set at 0.25 originally, this failed essentially
#: every cheap answer and would have reported a near-zero pass rate that measured
#: verbosity rather than quality. Only catastrophic brevity is treated as a defect.
MECHANICAL_LENGTH_FLOOR = 0.05

_JUDGE_PROMPT = """You are grading whether a cheaper model's answer is good \
enough to ship in place of a stronger model's answer.

Original request:
{prompt}

Answer A (from the stronger, more expensive model):
{reference}

Answer B (from the cheaper model, the one being graded):
{candidate}

Grade Answer B on whether it would serve the user as well as Answer A. Judge \
correctness and completeness, not style or length -- a shorter answer that is \
correct and complete should score highly.

Reply in exactly this format, with no other text:
SCORE: <a number between 0.0 and 1.0>
REASON: <one sentence naming the most important difference>"""


@dataclass(frozen=True)
class Verdict:
    """The outcome of one quality comparison."""

    passed: bool
    score: float
    reason: str
    judge: str
    """Which judge produced this. Reported alongside any pass rate."""


@runtime_checkable
class Judge(Protocol):
    name: str

    async def grade(self, prompt: str, candidate: str, reference: str) -> Verdict: ...


class MechanicalJudge:
    """Checks what can be checked without understanding the answer.

    Catches the failures that need no judgment -- empty output, a stub, an answer
    a fraction of the reference's length -- and passes everything else. It will
    happily pass a confidently wrong answer, which is exactly why its verdicts
    are labelled.
    """

    name = "mechanical"

    def __init__(self, length_floor: float = MECHANICAL_LENGTH_FLOOR) -> None:
        self._length_floor = length_floor

    async def grade(self, prompt: str, candidate: str, reference: str) -> Verdict:
        candidate = candidate.strip()
        reference = reference.strip()

        if not candidate:
            return self._fail(0.0, "cheap model returned nothing")
        if not reference:
            # Nothing to compare against; refuse to claim a pass.
            return self._fail(0.0, "reference model returned nothing to compare against")

        ratio = len(candidate) / len(reference)
        if ratio < self._length_floor:
            return self._fail(
                round(ratio, 3),
                f"cheap answer is {ratio:.0%} the length of the reference, "
                "which suggests it stopped early",
            )

        score = min(1.0, round(ratio, 3))
        return Verdict(
            passed=True,
            score=score,
            reason="no mechanical defect found (this judge cannot assess correctness)",
            judge=self.name,
        )

    def _fail(self, score: float, reason: str) -> Verdict:
        return Verdict(passed=False, score=score, reason=reason, judge=self.name)


class LLMJudge:
    """Asks a strong model to grade the cheap model's answer.

    A judge failure never fails a user's request -- the request has already been
    answered by the time this runs. An unparseable grade is reported as such
    rather than silently counted as a pass.
    """

    name = "llm"

    def __init__(
        self,
        model: Model,
        provider: Provider,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        max_output_tokens: int = 512,
    ) -> None:
        self._model = model
        self._provider = provider
        self._threshold = pass_threshold
        self._max_output_tokens = max_output_tokens

    async def grade(self, prompt: str, candidate: str, reference: str) -> Verdict:
        graded = _JUDGE_PROMPT.format(
            prompt=prompt, reference=reference, candidate=candidate
        )
        try:
            completion = await self._provider.complete(
                model=self._model.name,
                messages=[Message(role="user", content=graded)],
                max_output_tokens=self._max_output_tokens,
            )
        except ProviderError as exc:
            return Verdict(
                passed=False,
                score=0.0,
                reason=f"judge unavailable: {exc}",
                judge=f"{self.name}:error",
            )

        score, reason = _parse_grade(completion.text)
        if score is None:
            return Verdict(
                passed=False,
                score=0.0,
                reason=f"judge returned an unparseable grade: {completion.text[:120]!r}",
                judge=f"{self.name}:unparseable",
            )

        return Verdict(
            passed=score >= self._threshold,
            score=score,
            reason=reason or "no reason given",
            judge=f"{self.name}:{self._model.name}",
        )


def _parse_grade(text: str) -> tuple[float | None, str]:
    """Pull SCORE and REASON out of a judge response."""
    # The minus sign is accepted so an out-of-range grade clamps to a clear
    # failure rather than being reported as an unparseable response.
    score_match = re.search(r"SCORE:\s*(-?[0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)

    score = None
    if score_match:
        try:
            score = max(0.0, min(1.0, float(score_match.group(1))))
        except ValueError:  # pragma: no cover - the regex already constrains this
            score = None

    reason = reason_match.group(1).strip() if reason_match else ""
    return score, reason
