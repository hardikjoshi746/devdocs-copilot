"""
evaluator/faithfulness_check.py

Post-generation faithfulness check — filters out ungrounded claims from the LLM answer.

Why this exists:
- Even with good retrieval, LLMs sometimes generate claims not in the context
- These "hallucinations" look credible but are wrong
- This check strips ungrounded claims before returning the answer to the user

How it works:
- Combines all retrieved chunks into one context block
- Asks Claude to rewrite the answer keeping only claims supported by the context
- Returns the filtered answer

This is a second independent check on the same property as the retrieval evaluator —
the evaluator checks chunks BEFORE generation, this checks the answer AFTER generation.
"""

from ingestion.chunkers import Document
from anthropic import AsyncAnthropic
import os


async def check_faithfulness(answer: str, docs: list[Document]) -> str:
    """
    Filter the generated answer to remove claims not grounded in retrieved chunks.

    Args:
        answer: the raw LLM-generated answer
        docs:   the retrieved chunks used as context for generation

    Returns:
        filtered answer with only claims supported by the context
    """
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Combine all chunks into one context block — Claude checks every claim
    # against the full set of retrieved chunks, not just one at a time
    context = "\n\n".join(doc.content for doc in docs)

    prompt = (
        f"You are given an answer and supporting context chunks.\n"
        f"Remove any claims from the answer that are not supported by the context.\n"
        f"Return only the filtered answer with grounded claims.\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context:\n{context}"
    )
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text