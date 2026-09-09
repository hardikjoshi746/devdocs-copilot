import hashlib
import json
from redis.asyncio import Redis


def _key(question: str) -> str:
    normalized = question.lower().strip()
    return "answer:" + hashlib.sha256(normalized.encode()).hexdigest()


async def get_cached(redis: Redis, question: str) -> dict | None:
    raw = await redis.get(_key(question))
    return json.loads(raw) if raw else None


async def set_cached(redis: Redis, question: str, response: dict) -> None:
    # No TTL — entries live until re-ingestion flushes the cache or
    # Redis evicts under memory pressure (allkeys-lru policy)
    await redis.set(_key(question), json.dumps(response))