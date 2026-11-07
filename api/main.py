import json
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

import jwt as pyjwt
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from redis.asyncio import Redis

from query.pipeline import pipeline
from evaluator.retrieval_evaluator import evaluate
from generation.answer import generate
from evaluator.faithfulness_check import check_faithfulness
from ingestion.chunkers import Document
from monitoring.tracer import trace, span
from monitoring.logger import log
from api.cache import get_cached, set_cached
from api.guardrails import validate_input, mask_pii, canary_prompt_fragment, check_canary_leak

load_dotenv()

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieval_quality: str


# ---------------------------------------------------------------------------
# App lifecycle — Redis connection
# ---------------------------------------------------------------------------

_redis: Redis | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    _redis = Redis.from_url(redis_url, decode_responses=True)
    yield
    await _redis.aclose()

app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Parent-chunk index (built once at startup)
# ---------------------------------------------------------------------------

_chunk_index: dict[str, Document] = {}
_chunks_path = Path("data/chunks/chunks.jsonl")
if _chunks_path.exists():
    with _chunks_path.open() as f:
        for line in f:
            chunk = Document(**json.loads(line))
            _chunk_index[chunk.id] = chunk


def _load_parent_chunks(docs: list[Document]) -> list[Document]:
    existing_ids = {doc.id for doc in docs}
    parents = [
        _chunk_index[doc.parent_id]
        for doc in docs
        if doc.parent_id and doc.parent_id in _chunk_index and doc.parent_id not in existing_ids
    ]
    return docs + parents


# ---------------------------------------------------------------------------
# Authentication — JWT Bearer tokens
#
# API keys are signed JWTs issued offline (see scripts/issue_token.py).
# Each token carries a `repos` claim listing which repos the holder may query.
# The signing secret (JWT_SIGNING_SECRET) is the only thing that must stay
# secret — knowing a repo name is not enough to forge a valid token.
#
# .env:
#   JWT_SIGNING_SECRET=<random 32+ byte hex string>
#
# Token structure:
#   {"sub": "tenant_id", "repos": ["fastapi", "django"], "exp": <unix ts>}
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> tuple[str, list[str]]:
    """
    Verify the Bearer JWT and return (sub, allowed_repos).
    Raises HTTP 401 on missing / invalid / expired tokens.
    Raises HTTP 503 if JWT_SIGNING_SECRET is not configured (fail closed).
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required (Bearer <token>)",
        )

    secret = os.environ.get("JWT_SIGNING_SECRET")
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="Server is not configured to accept requests (JWT_SIGNING_SECRET missing)",
        )

    try:
        payload = pyjwt.decode(
            credentials.credentials,
            secret,
            algorithms=["HS256"],
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    sub = payload.get("sub", "unknown")
    repos = payload.get("repos", [])

    if not isinstance(repos, list) or not all(isinstance(r, str) for r in repos):
        raise HTTPException(status_code=401, detail="Invalid repos claim in token")

    return sub, repos


# ---------------------------------------------------------------------------
# Rate limiting — Redis token bucket (Lua script, atomic)
#
# Capacity  : 10 tokens
# Refill rate: 1 token / 6 seconds  (= 10 / minute at steady state)
# Burst      : up to 10 requests instantly, then 1 per 6s thereafter
#
# Key: tokenbucket:{sub}  (per-tenant, not per-IP)
# ---------------------------------------------------------------------------

_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity    = tonumber(ARGV[1])
local rate        = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens      = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
    tokens      = capacity
    last_refill = now
end

local elapsed   = math.max(0, now - last_refill)
local new_tokens = math.min(capacity, tokens + elapsed * rate)

local allowed
if new_tokens >= 1.0 then
    new_tokens = new_tokens - 1.0
    allowed = 1
else
    allowed = 0
end

redis.call('HMSET', key, 'tokens', tostring(new_tokens), 'last_refill', tostring(now))
redis.call('EXPIRE', key, 3600)
return allowed
"""

_BUCKET_CAPACITY = 10
_BUCKET_RATE = 1 / 6  # tokens per second


async def _check_rate_limit(sub: str, redis: Redis) -> None:
    allowed = await redis.eval(
        _TOKEN_BUCKET_LUA,
        1,
        f"tokenbucket:{sub}",
        _BUCKET_CAPACITY,
        _BUCKET_RATE,
        time.time(),
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": "60"},
            detail="Rate limit exceeded: 10 requests/minute",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    auth: tuple[str, list[str]] = Depends(verify_token),
):
    sub, allowed_repos = auth

    # 1. Validate and sanitize input (normalize + length cap + regex + LLM classifier)
    validated_question = await validate_input(request.question)

    # 2. Rate limiting — token bucket per tenant
    if _redis:
        await _check_rate_limit(sub, _redis)

    # 3. Cache check — keyed by (question, repos) to prevent cross-tenant hits
    if _redis:
        cached = await get_cached(_redis, validated_question, allowed_repos)
        if cached:
            log("cache_hit", question=validated_question)
            return QueryResponse(**cached)

    with trace(validated_question) as trace_id:
        docs = await pipeline(validated_question, allowed_repos=allowed_repos, trace_id=trace_id)

        with span(trace_id, "evaluate") as s:
            quality = await evaluate(validated_question, docs)
            s["metadata"]["quality"] = quality
            log("query_complete", trace_id=trace_id, quality=quality, question=validated_question)

        if quality == "ABSTAIN":
            log("query_abstain", trace_id=trace_id, question=validated_question)
            return JSONResponse(status_code=503, content={"error": "retrieval_quality_too_low"})

        if quality == "EXPAND":
            docs = _load_parent_chunks(docs)

        with span(trace_id, "generate"):
            result = await generate(
                validated_question, docs,
                canary=canary_prompt_fragment(),
            )
            log("tokens_used", trace_id=trace_id, question=validated_question, **result["usage"])

        # Canary check — if the model echoed the system-prompt secret, a poisoned
        # chunk triggered an exfiltration attempt. Return 503 (not 400) so the
        # caller learns nothing about the detection mechanism.
        if check_canary_leak(result["answer"]):
            log("canary_leak_detected", trace_id=trace_id, question=validated_question)
            return JSONResponse(status_code=503, content={"error": "retrieval_quality_too_low"})

        with span(trace_id, "check_faithfulness"):
            filtered_answer = await check_faithfulness(result["answer"], docs)

    # 4. Mask PII in the final answer before returning to caller
    safe_answer = mask_pii(filtered_answer)

    response = QueryResponse(
        answer=safe_answer,
        citations=result["citations"],
        retrieval_quality=quality,
    )

    if _redis:
        await set_cached(_redis, validated_question, allowed_repos, response.model_dump())

    return response