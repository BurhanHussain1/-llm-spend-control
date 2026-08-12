"""Central configuration for the gateway.

Every environment variable the project reads is declared here exactly once. No
other module should touch ``os.environ`` -- import :func:`get_settings` instead.

Two rules shape the defaults below:

* the gateway must start with no API keys and no external services, so a
  reviewer can clone the repo and run it immediately;
* switching to Postgres or Redis must be a config change, never a code change.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -------------------------------------------------------------
    database_url: str = f"sqlite:///{DATA_DIR / 'spend.db'}"
    redis_url: str | None = None
    """When unset, spend counters live in an in-memory dict (single process only)."""

    # --- providers -----------------------------------------------------------
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    enable_ollama: bool = False
    """Ollama needs no API key, so it cannot be detected the way the paid
    providers are -- it has to be turned on explicitly."""

    default_provider: str = "mock"
    """Provider used when a model has no credentials. Empty string = hard error."""

    request_timeout_seconds: float = 30.0

    # --- behaviour -----------------------------------------------------------
    budget_warn_threshold: float = 0.8
    """Fraction of budget that triggers a warning. Blocking always happens at 1.0."""

    shadow_sample_rate: float = 0.1
    """Fraction of cheap-model responses re-scored against the strongest model."""

    escalate_on_high_priority: bool = True

    def counters_backend(self) -> str:
        """Return which spend-counter backend these settings select."""
        return "redis" if self.redis_url else "memory"

    def ensure_directories(self) -> None:
        """Create the local folders the defaults assume exist.

        Called on startup and by scripts. Safe to call repeatedly.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built once and reused.

    Tests that change the environment must call ``get_settings.cache_clear()``.
    """
    return Settings()
