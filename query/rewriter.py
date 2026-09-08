"""
query/rewriter.py

HyDE (Hypothetical Document Embeddings) query rewriter.

Problem: "how do I handle a 404?" is semantically distant from
"raise HTTPException(status_code=404, detail='Not found')" even though
that's exactly the answer. The embedding vectors are far apart.

HyDE solution:
1. Generate a fake "ideal answer" using an LLM
2. Embed the fake answer instead of the original question
3. The fake answer uses the same vocabulary as real docs → better vector matches
4. Throw away the fake answer after retrieval — only used for finding chunks

Why claude-haiku-4-5: fast and cheap — HyDE runs on every query so cost matters.
Max 300 tokens — we only need a short answer to get good embeddings, not a full response.
"""

from anthropic import AsyncAnthropic
import os
from dotenv import load_dotenv

load_dotenv()
_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
_cache: dict[str, str] = {}

async def rewrite(query: str) -> str:
    """
    Generate a hypothetical answer to the query for use as a retrieval vector.

    The returned text is NOT shown to the user — it's only used to embed
    and find similar chunks in Chroma. The original query is still used
    for BM25 sparse search (exact keyword matching).
    """
    if query in _cache:
        return _cache[query]
    prompt = f"Write a short technical answer to this question as if answering from FastAPI documentation: {query}"
    response = await _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    _cache[query] = response.content[0].text
    return _cache[query]