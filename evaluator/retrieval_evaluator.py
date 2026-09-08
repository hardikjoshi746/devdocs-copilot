"""
evaluator/retrieval_evaluator.py

Scores retrieved chunks for relevance and routes to GOOD / EXPAND / ABSTAIN.

Why this exists:
- After reranking we have top-5 chunks, but they might still be irrelevant
- Passing bad context to the LLM causes confident hallucinations
- Better to return a 503 (ABSTAIN) than a wrong answer with citations

Routing logic:
- GOOD   (avg >= 0.7): chunks are relevant, proceed to LLM generation
- EXPAND (avg 0.4-0.7): borderline — fetch parent chunks, broaden query, retry once
- ABSTAIN (avg < 0.4): chunks are irrelevant — return 503, log trace

Why claude-haiku-4-5 for scoring:
- Fast and cheap — runs on every query for every chunk
- Scores all chunks concurrently via asyncio.gather
- max_tokens=10 — we only need a single number back
"""

from ingestion.chunkers import Document
from anthropic import AsyncAnthropic
import os
import asyncio
import re


async def _score_doc(client: AsyncAnthropic, query: str, doc: Document) -> float:
    """
    Ask Claude to rate how relevant a single chunk is for answering the query.
    Returns a float between 0 and 1.

    Uses regex to extract the score in case Claude adds extra text despite
    being told to reply with only a number.
    """
    prompt = (
        f"Rate how relevant this chunk is for answering the query. "
        f"Reply with ONLY a number between 0 and 1.\n\n"
        f"Query: {query}\n\nChunk: {doc.content}"
    )
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    # regex extracts first number matching 0, 1, or 0.xxx from response
    match = re.search(r"[01](\.\d+)?", text)
    return float(match.group()) if match else 0.0


async def evaluate(query: str, docs: list[Document]) -> str:
    """
    Score all chunks concurrently and return the routing decision.

    All chunk scores are fetched in parallel via asyncio.gather —
    5 API calls take the same time as 1 instead of 5x sequential latency.
    """
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Fire all scoring calls concurrently — chunks are independent of each other
    scores = await asyncio.gather(*[_score_doc(client, query, doc) for doc in docs])

    avg = sum(scores) / len(scores)

    if avg >= 0.7:
        return "GOOD"
    elif avg >= 0.4:
        return "EXPAND"
    else:
        return "ABSTAIN"