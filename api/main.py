from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from query.pipeline import pipeline
from evaluator.retrieval_evaluator import evaluate
from generation.answer import generate
from evaluator.faithfulness_check import check_faithfulness

load_dotenv()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]   
    retrieval_quality: str

app = FastAPI()

@app.get("/health")
def healthCheck():
    return {"status" : "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    docs = await pipeline(request.question)
    quality = await evaluate(request.question, docs)
    if quality == "ABSTAIN":
        return JSONResponse(status_code=503, content={"error" : "retrieval_quality_too_low"})
    result = await generate(request.question, docs)
    filtered_answer = await check_faithfulness(result["answer"], docs)

    return QueryResponse(
        answer=filtered_answer,
        citations=result["citations"],
        retrieval_quality=quality
    )