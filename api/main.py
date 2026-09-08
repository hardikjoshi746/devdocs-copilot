import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from query.pipeline import pipeline
from evaluator.retrieval_evaluator import evaluate
from generation.answer import generate
from evaluator.faithfulness_check import check_faithfulness
from ingestion.chunkers import Document
from monitoring.tracer import trace, span
from monitoring.logger import log

load_dotenv()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieval_quality: str

app = FastAPI()

# Built once at startup — O(1) parent chunk lookup at query time
_chunk_index: dict[str, Document] = {}
_chunks_path = Path("data/chunks/chunks.jsonl")
if _chunks_path.exists():
    with _chunks_path.open() as f:
        for line in f:
            chunk = Document(**json.loads(line))
            _chunk_index[chunk.id] = chunk


def _load_parent_chunks(docs: list[Document]) -> list[Document]:
    """
    For docs that have a parent_id, fetch the parent chunk from the index.
    Returns the original docs plus any parent chunks not already present.
    """
    existing_ids = {doc.id for doc in docs}
    parents = [
        _chunk_index[doc.parent_id]
        for doc in docs
        if doc.parent_id and doc.parent_id in _chunk_index and doc.parent_id not in existing_ids
    ]
    return docs + parents


@app.get("/health")
def healthCheck():
    return {"status" : "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    with trace(request.question) as trace_id:
        docs = await pipeline(request.question, trace_id=trace_id)

        with span(trace_id, "evaluate") as s:
            quality = await evaluate(request.question, docs)
            s["metadata"]["quality"] = quality
            log("query_complete", trace_id=trace_id, quality=quality, question=request.question)

        if quality == "ABSTAIN":
            log("query_abstain", trace_id=trace_id, question=request.question)
            return JSONResponse(status_code=503, content={"error" : "retrieval_quality_too_low"})

        if quality == "EXPAND":
            # Fetch parent chunks to broaden context for borderline retrievals
            docs = _load_parent_chunks(docs)

        with span(trace_id, "generate"):
            result = await generate(request.question, docs)
        with span(trace_id, "check_faithfulness"):
            filtered_answer = await check_faithfulness(result["answer"], docs)

        return QueryResponse(
            answer=filtered_answer,
            citations=result["citations"],
            retrieval_quality=quality
        )