"""Budget limits, loaded from ``config/budgets.yaml``.

Answers one question: for this team and feature, what limits apply? A feature
limit is additional to its team's limit rather than a replacement, so a single
runaway feature cannot spend past its team's cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.settings import CONFIG_DIR


class PolicyError(ValueError):
    """The budgets file is missing or malformed."""


@dataclass(frozen=True)
class Limits:
    """A daily and monthly ceiling for one scope."""

    daily_limit_usd: float
    monthly_limit_usd: float

    def __post_init__(self) -> None:
        if self.daily_limit_usd <= 0 or self.monthly_limit_usd <= 0:
            raise PolicyError("budget limits must be positive")


class BudgetPolicies:
    """Validated team and feature limits."""

    def __init__(
        self,
        defaults: Limits,
        teams: dict[str, Limits],
        features: dict[tuple[str, str], Limits],
    ) -> None:
        self._defaults = defaults
        self._teams = teams
        self._features = features

    @classmethod
    def load(cls, path: Path | None = None) -> "BudgetPolicies":
        path = path or CONFIG_DIR / "budgets.yaml"
        if not path.exists():
            raise PolicyError(f"budgets file not found: {path}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        defaults_raw = raw.get("defaults")
        if not defaults_raw:
            raise PolicyError(f"{path}: 'defaults' is required")
        defaults = _parse_limits(defaults_raw, "defaults", fallback=None)

        teams: dict[str, Limits] = {}
        features: dict[tuple[str, str], Limits] = {}

        for team_id, team_raw in (raw.get("teams") or {}).items():
            team_raw = team_raw or {}
            team_limits = _parse_limits(team_raw, team_id, fallback=defaults)
            teams[team_id] = team_limits

            for feature, feature_raw in (team_raw.get("features") or {}).items():
                features[(team_id, feature)] = _parse_limits(
                    feature_raw or {},
                    f"{team_id}.{feature}",
                    fallback=team_limits,
                )

        return cls(defaults=defaults, teams=teams, features=features)

    def for_team(self, team_id: str) -> Limits:
        """Limits for a team. Unknown teams get the defaults rather than an error.

        An unrecognised team is far more likely to be a new product surface than
        an attack, and failing the request would be worse than capping it low.
        """
        return self._teams.get(team_id, self._defaults)

    def for_feature(self, team_id: str, feature: str) -> Limits | None:
        """Limits for a feature, or None when the feature has no cap of its own."""
        return self._features.get((team_id, feature))

    def known_teams(self) -> list[str]:
        return sorted(self._teams)


def _parse_limits(raw: dict, label: str, fallback: Limits | None) -> Limits:
    """Read a limits block, inheriting anything it omits from `fallback`."""
    if not isinstance(raw, dict):
        raise PolicyError(f"{label}: expected a mapping, got {type(raw)}")

    daily = raw.get("daily_limit_usd")
    monthly = raw.get("monthly_limit_usd")

    if daily is None:
        if fallback is None:
            raise PolicyError(f"{label}: 'daily_limit_usd' is required")
        daily = fallback.daily_limit_usd
    if monthly is None:
        if fallback is None:
            raise PolicyError(f"{label}: 'monthly_limit_usd' is required")
        monthly = fallback.monthly_limit_usd

    try:
        return Limits(daily_limit_usd=float(daily), monthly_limit_usd=float(monthly))
    except PolicyError as exc:
        raise PolicyError(f"{label}: {exc}") from exc
