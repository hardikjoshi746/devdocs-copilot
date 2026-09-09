import hashlib
import json
import os
from redis.asyncio import Redis

TTL = int(os.environ.get("CACHE_TTL_SECONDS", 3600))


def _key(question: str) -> str:
    normalized = question.lower().strip()
    return "answer:" + hashlib.sha256(normalized.encode()).hexdigest()


async def get_cached(redis: Redis, question: str) -> dict | None:
    raw = await redis.get(_key(question))
    return json.loads(raw) if raw else None


async def set_cached(redis: Redis, question: str, response: dict) -> None:
    await redis.setex(_key(question), TTL, json.dumps(response))