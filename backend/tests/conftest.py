"""
Shared pytest fixtures.

The API test suite talks to a real database (MYSQL_URL), but the FastAPI
app's table-creation logic (`init_db()`) normally only runs inside the
app's `lifespan` context -- and httpx's `ASGITransport`, as used in this
test suite's `client` fixture, does NOT trigger FastAPI's startup/shutdown
lifespan events by default. Without this fixture, every test run against a
fresh database (exactly what happens in CI, with a brand-new ephemeral
MySQL container) fails with "Table 'health_platform.users' doesn't exist" --
not because anything is broken, but because nothing ever created the tables.

This fixture creates the schema once per test session, directly, so tests
don't depend on lifespan behavior that this test client doesn't trigger.
"""

import asyncio

import pytest

from app.db.database import engine, init_db


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    asyncio.run(init_db())
    asyncio.run(engine.dispose())
