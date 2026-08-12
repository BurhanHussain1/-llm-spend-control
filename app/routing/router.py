"""Model selection.

Takes a request and returns the model that will serve it, plus the reason. The
resolution order is fixed and each step can only raise the tier, never lower it:

1. **Caller preference** -- an explicit ``preferred_model`` wins outright.
2. **Feature override** -- some product surfaces always get a strong model.
3. **Risk tags** -- a tag applies a floor.
4. **Classifier** -- the fallback when nothing above applies.

Then, within the chosen tier, the first model that is reachable *and* has a big
enough context window is selected. If no model in that tier qualifies, the router
climbs to a stronger tier rather than quietly serving the request with something
too small.

The classifier is the weakest link in this chain -- see the holdout numbers in
``eval/classifier_eval.py``. The overrides above it exist precisely because a
text-based guess should not be the only thing standing between a legal question
and the cheapest available model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.registry import Model, ModelRegistry
from app.routing.classifier import Classification, classify
from app.schemas import ChatRequest
from app.settings import CONFIG_DIR

MAX_TIER = 3


class RoutingError(RuntimeError):
    """No model can serve this request."""


@dataclass(frozen=True)
class RoutingDecision:
    """The chosen model and a human-readable account of why."""

    model: Model
    tier: int
    reason: str
    classification: Classification | None = None
    """None when an override or caller preference made the classifier moot."""


class RoutingPolicy:
    """Tier-to-model preferences and override rules, from ``config/routing.yaml``."""

    def __init__(
        self,
        tiers: dict[int, list[str]],
        feature_overrides: dict[str, int],
        risk_tag_overrides: dict[str, int],
    ) -> None:
        self.tiers = tiers
        self.feature_overrides = feature_overrides
        self.risk_tag_overrides = risk_tag_overrides

    @classmethod
    def load(cls, path: Path | None = None) -> "RoutingPolicy":
        path = path or CONFIG_DIR / "routing.yaml"
        if not path.exists():
            raise RoutingError(f"routing config not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tiers_raw = raw.get("tiers") or {}

        tiers = {int(tier): list(models) for tier, models in tiers_raw.items()}
        for tier in (1, 2, 3):
            if not tiers.get(tier):
                raise RoutingError(f"{path}: tier {tier} has no models")

        return cls(
            tiers=tiers,
            feature_overrides={
                str(k): int(v) for k, v in (raw.get("feature_overrides") or {}).items()
            },
            risk_tag_overrides={
                str(k): int(v) for k, v in (raw.get("risk_tag_overrides") or {}).items()
            },
        )


class Router:
    """Resolves a request to a concrete model."""

    def __init__(
        self,
        registry: ModelRegistry,
        policy: RoutingPolicy,
        available_providers: set[str],
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._available = available_providers

    def route(self, request: ChatRequest, input_tokens: int = 0) -> RoutingDecision:
        """Choose a model for `request`.

        `input_tokens` lets the router reject models whose context window is too
        small. Passing 0 skips that check.
        """
        if request.preferred_model:
            return self._honour_preference(request, input_tokens)

        tier, reason, classification = self._resolve_tier(request)
        model, note = self._pick_model(tier, input_tokens)

        return RoutingDecision(
            model=model,
            tier=model.tier,
            reason=f"{reason}{note}",
            classification=classification,
        )

    # --- tier resolution -----------------------------------------------------

    def _resolve_tier(
        self, request: ChatRequest
    ) -> tuple[int, str, Classification | None]:
        """Return the required tier, why, and the classification if it was used."""
        feature_floor = self._policy.feature_overrides.get(request.feature)
        risk_floors = {
            tag: self._policy.risk_tag_overrides[tag]
            for tag in request.risk_tags
            if tag in self._policy.risk_tag_overrides
        }

        # A feature override at the top tier settles it -- no need to classify.
        if feature_floor == MAX_TIER:
            return (
                MAX_TIER,
                f"feature override: {request.feature!r} always uses tier {MAX_TIER}",
                None,
            )

        classification = classify(request.prompt_text(), request.risk_tags)
        tier = classification.tier
        reason = f"classifier: {classification.summary}"

        if feature_floor and feature_floor > tier:
            tier = feature_floor
            reason = (
                f"feature override: {request.feature!r} has a tier-{feature_floor} "
                f"floor (classifier said tier {classification.tier})"
            )

        if risk_floors:
            highest_tag, highest = max(risk_floors.items(), key=lambda item: item[1])
            if highest > tier:
                tier = highest
                reason = (
                    f"risk tag {highest_tag!r} has a tier-{highest} floor "
                    f"(classifier said tier {classification.tier})"
                )

        return tier, reason, classification

    # --- model selection -----------------------------------------------------

    def _pick_model(self, tier: int, input_tokens: int) -> tuple[Model, str]:
        """First reachable model in `tier` with room for the prompt.

        Climbs to stronger tiers when nothing in this one qualifies, because
        serving a request with a model whose context window is too small would
        truncate it.
        """
        attempted: list[str] = []

        for candidate_tier in range(tier, MAX_TIER + 1):
            for name in self._policy.tiers.get(candidate_tier, []):
                model = self._registry.get(name)

                if model.provider not in self._available:
                    attempted.append(f"{name} (provider unavailable)")
                    continue
                if input_tokens and input_tokens > model.max_context_tokens:
                    attempted.append(f"{name} (context too small)")
                    continue

                note = ""
                if candidate_tier != tier:
                    note = (
                        f"; upgraded from tier {tier} to {candidate_tier} because no "
                        f"tier-{tier} model qualified"
                    )
                return model, note

        raise RoutingError(
            f"no model available at tier {tier} or above. Tried: "
            f"{', '.join(attempted) or 'nothing'}. "
            f"Reachable providers: {sorted(self._available) or 'none'}"
        )

    def _honour_preference(
        self, request: ChatRequest, input_tokens: int
    ) -> RoutingDecision:
        """Use the caller's chosen model, or fail loudly.

        A request that names a model and silently gets a different one is worse
        than an error: the caller would have no idea their preference was ignored.
        """
        name = request.preferred_model or ""
        model = self._registry.get(name)  # raises RegistryError if unknown

        if model.provider not in self._available:
            raise RoutingError(
                f"requested model {name!r} needs provider {model.provider!r}, "
                f"which is not configured. Reachable providers: "
                f"{sorted(self._available) or 'none'}"
            )
        if input_tokens and input_tokens > model.max_context_tokens:
            raise RoutingError(
                f"requested model {name!r} has a {model.max_context_tokens}-token "
                f"context window, but this prompt is {input_tokens} tokens"
            )

        return RoutingDecision(
            model=model,
            tier=model.tier,
            reason=f"caller requested {name!r} explicitly",
            classification=None,
        )
