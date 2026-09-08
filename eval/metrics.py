"""
eval/metrics.py

Evaluation metrics for the RAG pipeline.

Metrics:
1. Recall@5         — is the correct source chunk in the top-5 results?
2. Answer Correctness — LLM-as-judge scores answer vs reference (0-5)
3. Faithfulness     — % of claims in the answer grounded in retrieved chunks
4. Latency          — end-to-end response time in seconds

Usage:
    from eval.metrics import recall_at_5, answer_correctness, faithfulness, measure_latency
"""

import time
import asyncio
import os
from anthropic import AsyncAnthropic
from ingestion.chunkers import Document

_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Recall@5
# ---------------------------------------------------------------------------

def recall_at_5(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """
    Returns 1.0 if any expected source id appears in the retrieved ids, else 0.0.

    Why binary (not graded)?
    For a Q&A system, either the right chunk is in context or it isn't.
    Partial credit doesn't map to a meaningful user experience difference.

    Args:
        retrieved_ids:  list of chunk ids returned by the pipeline (top-5)
        expected_ids:   list of chunk ids that should contain the answer

    Returns:
        1.0 if any expected id is in retrieved_ids, else 0.0
    """
    if not expected_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    return 1.0 if any(eid in retrieved_set for eid in expected_ids) else 0.0


# ---------------------------------------------------------------------------
# Answer Correctness (LLM-as-judge)
# ---------------------------------------------------------------------------

async def answer_correctness(
    question: str,
    generated_answer: str,
    reference_answer: str,
) -> float:
    """
    Use Claude to score the generated answer against the reference (0-5).

    Why LLM-as-judge?
    - Exact string matching fails for paraphrases
    - BLEU/ROUGE don't capture factual correctness
    - LLM judges correlate well with human evaluation at lower cost

    Scoring rubric:
    5 — correct, complete, well-explained
    4 — correct but missing minor details
    3 — partially correct
    2 — mostly wrong but contains some truth
    1 — wrong
    0 — completely irrelevant or refused to answer

    Returns score normalized to 0-1 (divide by 5).
    """
    prompt = (
        f"Score the generated answer against the reference answer on a scale of 0-5.\n"
        f"Reply with ONLY a single integer (0, 1, 2, 3, 4, or 5).\n\n"
        f"Scoring rubric:\n"
        f"5 = correct, complete, well-explained\n"
        f"4 = correct but missing minor details\n"
        f"3 = partially correct\n"
        f"2 = mostly wrong but contains some truth\n"
        f"1 = wrong\n"
        f"0 = completely irrelevant\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"Generated answer: {generated_answer}"
    )

    response = await _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}]
    )

    import re
    text = response.content[0].text.strip()
    match = re.search(r"[0-5]", text)
    return int(match.group()) / 5.0 if match else 0.0


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

async def faithfulness(
    answer: str,
    docs: list[Document],
) -> float:
    """
    Score what fraction of claims in the answer are grounded in the retrieved chunks.

    Asks Claude to count total claims and grounded claims, returns the ratio.

    Why this matters:
    A high-quality retrieval system can still produce unfaithful answers if the
    LLM adds knowledge beyond the retrieved context. This metric catches that.

    Returns float between 0.0 and 1.0.
    """
    context = "\n\n".join(doc.content for doc in docs)

    prompt = (
        f"Count the total number of factual claims in the answer, "
        f"and how many of those claims are directly supported by the context.\n"
        f"Reply with ONLY two integers separated by a slash, like: 4/5\n"
        f"(supported_claims/total_claims)\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context:\n{context}"
    )

    response = await _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    import re
    text = response.content[0].text.strip()
    nums = re.findall(r"\d+", text)
    if len(nums) >= 2:
        supported, total = int(nums[0]), int(nums[1])
        return supported / total if total > 0 else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

async def measure_latency(question: str) -> float:
    """
    Measure end-to-end pipeline latency in seconds for a single question.

    Returns elapsed time in seconds.
    """
    from query.pipeline import pipeline
    from generation.answer import generate

    start = time.perf_counter()
    docs = await pipeline(question)
    await generate(question, docs)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Aggregate — run all metrics for one question
# ---------------------------------------------------------------------------

async def evaluate_one(
    question: str,
    expected_answer: str,
    expected_source_ids: list[str],
) -> dict:
    """
    Run the full pipeline for one question and compute all metrics.

    Returns a dict with all metric scores and the generated answer.
    """
    from query.pipeline import pipeline
    from generation.answer import generate
    from evaluator.faithfulness_check import check_faithfulness

    start = time.perf_counter()
    docs = await pipeline(question)
    result = await generate(question, docs)
    filtered = await check_faithfulness(result["answer"], docs)
    latency = time.perf_counter() - start

    retrieved_ids = [doc.id for doc in docs]

    # Run correctness and faithfulness concurrently
    correctness, faith_score = await asyncio.gather(
        answer_correctness(question, filtered, expected_answer),
        faithfulness(filtered, docs),
    )

    return {
        "question": question,
        "answer": filtered,
        "citations": result["citations"],
        "recall_at_5": recall_at_5(retrieved_ids, expected_source_ids),
        "answer_correctness": correctness,
        "faithfulness": faith_score,
        "latency_s": round(latency, 2),
    }