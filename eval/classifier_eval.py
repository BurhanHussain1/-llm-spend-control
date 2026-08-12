"""Score the complexity classifier against hand-labeled prompts.

Run it:

    python -m eval.classifier_eval

Without this, the routing story is unfalsifiable -- "we route simple work to
cheap models" would be an assertion with nothing behind it.

The report deliberately separates the two kinds of mistake, because they are not
equally bad:

* **Under-routing** (predicted tier < true tier) sends hard work to a weak model.
  The failure is a wrong answer, which is the expensive kind.
* **Over-routing** (predicted tier > true tier) sends easy work to a strong model.
  The failure is a larger bill, which is merely wasteful.

A classifier with 85% accuracy that never under-routes is better than one with
90% accuracy that under-routes the risky requests, and an accuracy number on its
own cannot tell those two apart.

**Two files, on purpose.** ``labeled_prompts.jsonl`` is the calibration set: the
scoring weights were tuned against it, so its accuracy is optimistic and cannot
be quoted as a generalization estimate. ``holdout_prompts.jsonl`` was written
afterwards, deliberately using different phrasing, and the weights were **not**
adjusted in response to its results. The holdout number is the honest one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.registry import ModelRegistry
from app.routing.classifier import classify

CALIBRATION_PATH = Path(__file__).parent / "labeled_prompts.jsonl"
HOLDOUT_PATH = Path(__file__).parent / "holdout_prompts.jsonl"

#: A nominal request shape, used to price misroutes in dollars. Real requests
#: vary, so treat the dollar figures as relative rather than absolute.
NOMINAL_INPUT_TOKENS = 600
NOMINAL_OUTPUT_TOKENS = 300

TIERS = (1, 2, 3)


@dataclass
class Case:
    prompt: str
    true_tier: int
    predicted_tier: int
    score: float
    reasons: list[str]

    @property
    def correct(self) -> bool:
        return self.true_tier == self.predicted_tier

    @property
    def under_routed(self) -> bool:
        return self.predicted_tier < self.true_tier

    @property
    def over_routed(self) -> bool:
        return self.predicted_tier > self.true_tier


def load_cases(path: Path) -> list[Case]:
    if not path.exists():
        raise SystemExit(f"labeled prompts not found: {path}")

    cases = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{LABELS_PATH}:{line_number}: {exc}") from exc

        result = classify(record["prompt"], record.get("risk_tags"))
        cases.append(
            Case(
                prompt=record["prompt"],
                true_tier=int(record["tier"]),
                predicted_tier=result.tier,
                score=result.score,
                reasons=result.reasons,
            )
        )
    return cases


def tier_costs(registry: ModelRegistry) -> dict[int, float]:
    """Cost of one nominal request on the cheapest paid model in each tier.

    Free local models are skipped: including a $0 model would make every
    over-route look free, which would defeat the point of the comparison.
    """
    costs = {}
    for tier in TIERS:
        paid = [model for model in registry.for_tier(tier) if not model.is_free]
        cheapest = min(paid, key=lambda model: model.input_cost_per_mtok)
        costs[tier] = cheapest.cost(NOMINAL_INPUT_TOKENS, NOMINAL_OUTPUT_TOKENS)
    return costs


def report(cases: list[Case], costs: dict[int, float], title: str) -> None:
    total = len(cases)
    correct = sum(case.correct for case in cases)
    under = [case for case in cases if case.under_routed]
    over = [case for case in cases if case.over_routed]

    print(f"\n{title} -- {total} labeled prompts")
    print("=" * 62)
    print(f"accuracy        {correct / total:6.1%}  ({correct}/{total})")
    print(f"under-routed    {len(under) / total:6.1%}  ({len(under)})  <- quality risk")
    print(f"over-routed     {len(over) / total:6.1%}  ({len(over)})  <- wasted spend")

    print("\nConfusion matrix (rows = true tier, columns = predicted)")
    print("        " + "".join(f"  pred {tier}" for tier in TIERS) + "   recall")
    matrix = Counter((case.true_tier, case.predicted_tier) for case in cases)
    for true_tier in TIERS:
        row = [matrix[(true_tier, predicted)] for predicted in TIERS]
        support = sum(row)
        recall = row[true_tier - 1] / support if support else 0.0
        print(
            f" true {true_tier} " + "".join(f"{count:8d}" for count in row) + f"{recall:9.1%}"
        )

    print("\nPer-tier precision")
    for tier in TIERS:
        predicted = sum(1 for case in cases if case.predicted_tier == tier)
        hit = sum(1 for case in cases if case.predicted_tier == tier and case.correct)
        precision = hit / predicted if predicted else 0.0
        print(f"  tier {tier}: {precision:6.1%}  ({hit}/{predicted} predictions correct)")

    print("\nCost of mistakes, on a nominal "
          f"{NOMINAL_INPUT_TOKENS}-in / {NOMINAL_OUTPUT_TOKENS}-out request")
    wasted = sum(costs[case.predicted_tier] - costs[case.true_tier] for case in over)
    print(f"  over-routing wastes  ${wasted:.4f} across {len(over)} requests")
    print(f"  under-routing risks a wrong answer on {len(under)} requests")
    print("  (the second number is the one that matters -- money is recoverable,")
    print("   a wrong answer shipped to a customer is not)")

    if under:
        print(f"\nUnder-routed prompts ({len(under)}) -- fix these first")
        for case in under:
            print(f"  [true {case.true_tier} -> got {case.predicted_tier}] {case.prompt[:72]}")
            print(f"      score {case.score:+.1f}: {'; '.join(case.reasons) or 'no signals'}")

    if over:
        print(f"\nOver-routed prompts ({len(over)})")
        for case in over:
            print(f"  [true {case.true_tier} -> got {case.predicted_tier}] {case.prompt[:72]}")
            print(f"      score {case.score:+.1f}: {'; '.join(case.reasons) or 'no signals'}")

    print()


def main() -> None:
    costs = tier_costs(ModelRegistry.load())

    report(
        load_cases(CALIBRATION_PATH),
        costs,
        "CALIBRATION SET (weights were tuned on this -- optimistic)",
    )
    report(
        load_cases(HOLDOUT_PATH),
        costs,
        "HOLDOUT SET (never tuned against -- this is the honest number)",
    )
    print(
        "Quote the holdout accuracy, not the calibration accuracy. The weights\n"
        "were fitted to the calibration prompts after inspecting their errors, so\n"
        "that score measures fit, not generalization.\n"
    )


if __name__ == "__main__":
    main()
