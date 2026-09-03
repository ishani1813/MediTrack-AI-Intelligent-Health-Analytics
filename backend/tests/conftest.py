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

Root-cause fix for the event-loop bug (previously worked around by skipping
test_login_wrong_password): `engine` in app/db/database.py is a module-level
singleton. Its connection pool binds to whichever asyncio event loop is
running the first time a connection is actually opened. Test frameworks
default to giving each async test function its own fresh event loop, so the
engine would end up bound to test A's loop, then get invalidly reused --
from a different loop -- by test B, causing intermittent
"Task <Task ...> attached to a different loop" failures (confirmed
reproducible: 3 local runs against a real MySQL-compatible database, run
before this fix, all hit it). The actual fix is for every test in the
session to share the SAME event loop, so the engine is always used from the
loop it was created in. `asyncio.run()` was also part of the problem here:
it creates and destroys its own throwaway loop on every call, so even the
schema-creation step below used to run in a *different* loop than the tests
that followed it.
"""

import asyncio

import pytest
import pytest_asyncio

from app.db.database import engine, init_db


@pytest.fixture(scope="session")
def event_loop():
    """One event loop shared by every test in the session (see module
    docstring). Without this, pytest-asyncio's default is a fresh loop per
    test function, which is exactly what causes the cross-loop bug.

    pytest-asyncio 0.23.7 (the version pinned in requirements.txt) prints a
    DeprecationWarning for this pattern, pointing at a newer per-fixture
    `loop_scope` parameter -- but that parameter doesn't exist yet in 0.23.x,
    it was added in a later release. This is the correct, working approach
    for the version actually pinned here; revisit if that pin is upgraded.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _create_schema(event_loop):
    await init_db()
    yield
    await engine.dispose()
