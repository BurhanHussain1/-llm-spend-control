"""The model registry: what we can route to, and what it costs.

Loads ``config/models.yaml`` and answers three questions for the rest of the app:

* what models exist, and what are their limits and capabilities
* what does a given (model, input tokens, output tokens) triple cost
* which models serve a given routing tier

Everything here is pure -- no network, no database. That makes the cost
arithmetic, which every reported dollar figure depends on, trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from app.settings import CONFIG_DIR

TOKENS_PER_MILLION = 1_000_000

#: Costs are rounded to this many decimal places. A single request can easily
#: cost less than a thousandth of a cent, so 8 places keeps small requests from
#: rounding to zero while stopping float noise from leaking into stored totals.
COST_PRECISION = 8

VALID_TIERS = (1, 2, 3)


class RegistryError(ValueError):
    """The registry file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Model:
    """One routable model and everything the gateway needs to know about it."""

    name: str
    provider: str
    tier: int
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    typical_latency_ms: int
    max_context_tokens: int
    supports_vision: bool
    supports_tools: bool

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the dollar cost of a call with these token counts.

        Worked example: ``claude-haiku-4-5`` at $1.00 in / $5.00 out per million
        tokens, with 1,000 input and 500 output tokens::

            1000/1e6 * 1.00  +  500/1e6 * 5.00  =  0.001 + 0.0025  =  0.0035
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts cannot be negative")

        total = (
            input_tokens / TOKENS_PER_MILLION * self.input_cost_per_mtok
            + output_tokens / TOKENS_PER_MILLION * self.output_cost_per_mtok
        )
        return round(total, COST_PRECISION)

    @property
    def is_free(self) -> bool:
        """True for locally hosted models, which carry no per-token bill."""
        return self.input_cost_per_mtok == 0.0 and self.output_cost_per_mtok == 0.0


class ModelRegistry:
    """An immutable, validated view of ``config/models.yaml``."""

    def __init__(
        self,
        models: Iterable[Model],
        baseline_model: str,
        pricing_as_of: str,
    ) -> None:
        self._models = {model.name: model for model in models}
        self.pricing_as_of = pricing_as_of

        if not self._models:
            raise RegistryError("registry contains no models")
        if baseline_model not in self._models:
            raise RegistryError(
                f"baseline_model {baseline_model!r} is not one of the defined models"
            )
        self.baseline_model = baseline_model

    # --- loading -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelRegistry":
        """Read and validate the registry file.

        Raises :class:`RegistryError` with a message naming the offending model,
        so a typo in the YAML fails loudly at startup rather than silently
        mispricing requests later.
        """
        path = path or CONFIG_DIR / "models.yaml"
        if not path.exists():
            raise RegistryError(f"registry file not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = raw.get("models")
        if not isinstance(entries, list):
            raise RegistryError(f"{path}: expected a top-level 'models' list")

        models = [_parse_model(entry, index) for index, entry in enumerate(entries)]

        names = [model.name for model in models]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise RegistryError(f"duplicate model names: {sorted(duplicates)}")

        baseline = raw.get("baseline_model")
        if not baseline:
            raise RegistryError(f"{path}: 'baseline_model' is required")

        return cls(
            models=models,
            baseline_model=baseline,
            pricing_as_of=str(raw.get("pricing_as_of", "unknown")),
        )

    # --- lookups -------------------------------------------------------------

    def get(self, name: str) -> Model:
        try:
            return self._models[name]
        except KeyError:
            raise RegistryError(
                f"unknown model {name!r}; known models: {sorted(self._models)}"
            ) from None

    def all(self) -> list[Model]:
        return list(self._models.values())

    def names(self) -> list[str]:
        return sorted(self._models)

    def for_tier(self, tier: int) -> list[Model]:
        """Models serving `tier`, cheapest first.

        Ordering by input cost makes the router's "take the first affordable
        option" logic mean "take the cheapest option".
        """
        if tier not in VALID_TIERS:
            raise RegistryError(f"tier must be one of {VALID_TIERS}, got {tier!r}")
        return sorted(
            (model for model in self._models.values() if model.tier == tier),
            key=lambda model: (model.input_cost_per_mtok, model.output_cost_per_mtok),
        )

    def for_provider(self, provider: str) -> list[Model]:
        return [m for m in self._models.values() if m.provider == provider]

    def baseline(self) -> Model:
        """The model the savings report compares actual spend against."""
        return self.get(self.baseline_model)

    def strongest(self, among: Iterable[Model] | None = None) -> Model:
        """Return the highest-tier, most expensive model in `among`.

        Used to pick a shadow-verification and escalation target from whatever
        models are actually reachable, which is not always the configured
        baseline -- with no API keys set, only the mock models can run.
        """
        candidates = list(among) if among is not None else self.all()
        if not candidates:
            raise RegistryError("cannot pick the strongest model from an empty set")
        return max(
            candidates,
            key=lambda model: (model.tier, model.output_cost_per_mtok),
        )


def _parse_model(entry: object, index: int) -> Model:
    """Turn one YAML mapping into a validated :class:`Model`."""
    if not isinstance(entry, dict):
        raise RegistryError(f"models[{index}]: expected a mapping, got {type(entry)}")

    label = entry.get("name", f"models[{index}]")
    required = (
        "name",
        "provider",
        "tier",
        "input_cost_per_mtok",
        "output_cost_per_mtok",
        "typical_latency_ms",
        "max_context_tokens",
    )
    missing = [field for field in required if field not in entry]
    if missing:
        raise RegistryError(f"{label}: missing required fields {missing}")

    tier = entry["tier"]
    if tier not in VALID_TIERS:
        raise RegistryError(f"{label}: tier must be one of {VALID_TIERS}, got {tier!r}")

    for field in ("input_cost_per_mtok", "output_cost_per_mtok"):
        if entry[field] < 0:
            raise RegistryError(f"{label}: {field} cannot be negative")

    if entry["max_context_tokens"] <= 0:
        raise RegistryError(f"{label}: max_context_tokens must be positive")

    return Model(
        name=str(entry["name"]),
        provider=str(entry["provider"]),
        tier=int(tier),
        input_cost_per_mtok=float(entry["input_cost_per_mtok"]),
        output_cost_per_mtok=float(entry["output_cost_per_mtok"]),
        typical_latency_ms=int(entry["typical_latency_ms"]),
        max_context_tokens=int(entry["max_context_tokens"]),
        supports_vision=bool(entry.get("supports_vision", False)),
        supports_tools=bool(entry.get("supports_tools", False)),
    )
