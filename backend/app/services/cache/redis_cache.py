import json
import hashlib
from typing import Optional, Any
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logging import app_logger

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _redis_client


def make_cache_key(prefix: str, data: dict) -> str:
    """Create a deterministic cache key from prefix + sorted dict."""
    raw = json.dumps(data, sort_keys=True)
    digest = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{prefix}:{digest}"


async def cache_get(key: str) -> Optional[Any]:
    try:
        redis = await get_redis()
        value = await redis.get(key)
        if value:
            app_logger.debug(f"Cache HIT: {key}")
            return json.loads(value)
        app_logger.debug(f"Cache MISS: {key}")
        return None
    except Exception as e:
        app_logger.warning(f"Redis GET failed for {key}: {e}")
        return None


async def cache_set(key: str, value: Any, ttl: int = settings.CACHE_TTL_SECONDS) -> bool:
    try:
        redis = await get_redis()
        await redis.setex(key, ttl, json.dumps(value, default=str))
        app_logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
        return True
    except Exception as e:
        app_logger.warning(f"Redis SET failed for {key}: {e}")
        return False


async def cache_delete(key: str) -> bool:
    try:
        redis = await get_redis()
        await redis.delete(key)
        return True
    except Exception as e:
        app_logger.warning(f"Redis DELETE failed for {key}: {e}")
        return False


async def cache_invalidate_pattern(pattern: str) -> int:
    """Delete all keys matching a pattern (e.g. 'predict:patient_5:*')."""
    try:
        redis = await get_redis()
        keys = await redis.keys(pattern)
        if keys:
            await redis.delete(*keys)
        return len(keys)
    except Exception as e:
        app_logger.warning(f"Redis pattern delete failed for {pattern}: {e}")
        return 0
