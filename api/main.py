import json
import os
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI
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

load_dotenv()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieval_quality: str

# Redis client — shared across requests
_redis: Redis | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    _redis = Redis.from_url(redis_url, decode_responses=True)
    yield
    await _redis.aclose()

app = FastAPI(lifespan=lifespan)

# Built once at startup — O(1) parent chunk lookup at query time
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


@app.get("/health")
def healthCheck():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Cache check — skip entire pipeline on hit
    if _redis:
        cached = await get_cached(_redis, request.question)
        if cached:
            log("cache_hit", question=request.question)
            return QueryResponse(**cached)

    with trace(request.question) as trace_id:
        docs = await pipeline(request.question, trace_id=trace_id)

        with span(trace_id, "evaluate") as s:
            quality = await evaluate(request.question, docs)
            s["metadata"]["quality"] = quality
            log("query_complete", trace_id=trace_id, quality=quality, question=request.question)

        if quality == "ABSTAIN":
            log("query_abstain", trace_id=trace_id, question=request.question)
            return JSONResponse(status_code=503, content={"error": "retrieval_quality_too_low"})

        if quality == "EXPAND":
            docs = _load_parent_chunks(docs)

        with span(trace_id, "generate"):
            result = await generate(request.question, docs)
        with span(trace_id, "check_faithfulness"):
            filtered_answer = await check_faithfulness(result["answer"], docs)

    response = QueryResponse(
        answer=filtered_answer,
        citations=result["citations"],
        retrieval_quality=quality,
    )

    # Store in cache for next time
    if _redis:
        await set_cached(_redis, request.question, response.model_dump())

    return response