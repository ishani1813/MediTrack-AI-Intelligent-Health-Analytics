from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.MYSQL_URL,
    echo=settings.APP_ENV == "development",
    pool_size=10,
    max_overflow=20,
    # NOTE: pool_pre_ping=True is the more common way to guard against stale
    # connections, but it's broken with this exact SQLAlchemy 2.0.30 +
    # aiomysql 0.2.0 combination -- do_ping() calls dbapi_connection.ping()
    # with no arguments, while aiomysql's async adapter requires a
    # `reconnect` argument, raising a TypeError on every pooled checkout.
    # pool_recycle achieves the same goal (avoid handing out dead
    # connections) without hitting that code path: it proactively discards
    # any connection older than this many seconds instead of pinging it.
    # Tune this to just under your MySQL server's `wait_timeout`.
    pool_recycle=1800,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
