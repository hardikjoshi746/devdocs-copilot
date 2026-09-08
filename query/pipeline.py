"""
query/pipeline.py

Orchestrates the full retrieval pipeline for a single user query:

  query → HyDE rewrite → hybrid_search (dense + sparse + RRF) → rerank → top 5

Why this order?
1. rewrite:        generates a fake answer → better dense retrieval vector
2. hybrid_search:  dense (semantic) + sparse (BM25 keywords) → top 20 candidates
3. rerank:         cross-encoder scores (query, chunk) jointly → top 5 accurate results

Note: rewrite output is used for hybrid_search (both dense and sparse).
A future improvement: use original query for sparse, rewritten for dense only —
BM25 keyword matching works better with the original terse query.
"""

from ingestion.chunkers import Document
from query.rewriter import rewrite
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank_async
from monitoring.tracer import span


async def pipeline(query: str, trace_id: str = None) -> list[Document]:
    """
    Run the full retrieval pipeline and return top-5 most relevant chunks.

    Args:
        query: the raw user question

    Returns:
        top-5 Document chunks after rewriting, hybrid retrieval, and reranking
    """
    # Step 1: HyDE — generate a fake answer to use as the retrieval query
    # The fake answer uses FastAPI vocabulary, so it embeds closer to real chunks
    with span(trace_id, "rewrite"):
        hyde_response = await rewrite(query=query)

    # Step 2: Hybrid search — dense + BM25 + RRF fusion → top 20 candidates
    with span(trace_id, "hybrid_search"):
        response = await hybrid_search(query=hyde_response)

    # Step 3: Rerank — cross-encoder scores all 20 (query, chunk) pairs jointly → top 5
    # Uses original query here so reranker judges relevance to what user actually asked
    with span(trace_id, "rerank_async"):
        result = await rerank_async(query=query, docs=response)

    return result