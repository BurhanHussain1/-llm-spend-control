"""Database connection and session management.

One engine per process, built from ``DATABASE_URL``. The default is a local
SQLite file so the gateway starts with no infrastructure; pointing the same
variable at Postgres switches the storage layer with no code change, which is
what Docker Compose does.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.settings import get_settings


@lru_cache
def get_engine() -> Engine:
    """Return the process-wide engine, created once."""
    settings = get_settings()
    settings.ensure_directories()

    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        # FastAPI handles requests on a threadpool, and SQLite objects are
        # otherwise pinned to their creating thread.
        connect_args["check_same_thread"] = False

    return create_engine(
        settings.database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def create_tables() -> None:
    """Create any missing tables.

    Called on startup. Deliberately not Alembic: for a portfolio project a
    migration framework is ceremony, and the README lists it as a known gap
    rather than pretending it isn't one.
    """
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on failure.

    Used by scripts and background tasks. Request handlers get a session from
    the FastAPI dependency instead.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine so a new ``DATABASE_URL`` takes effect.

    Only for tests, which point each case at its own throwaway database.
    """
    get_engine.cache_clear()
    get_session_factory.cache_clear()
