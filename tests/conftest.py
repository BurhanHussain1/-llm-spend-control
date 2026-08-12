"""Shared fixtures.

Each test that touches the database gets its own throwaway SQLite file, so
tests never share state and never depend on a running Postgres.
"""

import pytest

from app.db.engine import create_tables, get_session_factory, reset_engine
from app.db.repository import Repository
from app.registry import ModelRegistry
from app.settings import get_settings


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    """A session against a fresh, empty database."""
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)

    get_settings.cache_clear()
    reset_engine()
    create_tables()

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
        reset_engine()
        get_settings.cache_clear()


@pytest.fixture
def repo(db_session):
    return Repository(db_session)


@pytest.fixture
def registry():
    return ModelRegistry.load()
