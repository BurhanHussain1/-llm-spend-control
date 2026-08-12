"""Settings behave as documented: safe defaults, and the environment wins."""

from app.settings import Settings, get_settings


def test_defaults_need_no_external_services() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url.startswith("sqlite:///")
    assert settings.redis_url is None
    assert settings.counters_backend() == "memory"
    assert settings.default_provider == "mock"


def test_defaults_match_the_documented_thresholds() -> None:
    settings = Settings(_env_file=None)

    assert settings.budget_warn_threshold == 0.8
    assert settings.escalate_on_high_priority is True

    # Low on purpose: a verification call costs roughly 25x the tier-1 request it
    # checks, so a high sample rate spends the saving on measuring it.
    assert settings.shadow_sample_rate == 0.02


def test_environment_overrides_defaults(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("BUDGET_WARN_THRESHOLD", "0.5")

    settings = Settings(_env_file=None)

    assert settings.counters_backend() == "redis"
    assert settings.budget_warn_threshold == 0.5


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
