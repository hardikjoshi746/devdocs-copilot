import hashlib
import json
from redis.asyncio import Redis


def _key(question: str, allowed_repos: list[str]) -> str:
    # Include repos in the key so tenants with different access scopes never
    # share a cached response. Without this, tenant A's answer (built from
    # repos=[fastapi]) could be served to tenant B with repos=[django].
    normalized = question.lower().strip()
    repos_str = ",".join(sorted(allowed_repos))
    return "answer:" + hashlib.sha256(f"{normalized}|{repos_str}".encode()).hexdigest()


async def get_cached(redis: Redis, question: str, allowed_repos: list[str]) -> dict | None:
    raw = await redis.get(_key(question, allowed_repos))
    return json.loads(raw) if raw else None


async def set_cached(redis: Redis, question: str, allowed_repos: list[str], response: dict) -> None:
    await redis.set(_key(question, allowed_repos), json.dumps(response))