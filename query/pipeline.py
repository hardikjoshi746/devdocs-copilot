"""
query/pipeline.py

Orchestrates the full retrieval pipeline for a single user query:

  query → route → HyDE rewrite → hybrid_search (dense + sparse + RRF) → rerank → top 5

Why this order?
1. route:          classifies query type (code/doc/issue/general) → filters retrieval
2. rewrite:        generates a fake answer → better dense retrieval vector
3. hybrid_search:  dense (semantic) + sparse (BM25 keywords) → top 20 candidates
4. rerank:         cross-encoder scores (query, chunk) jointly → top 5 accurate results

Note: rewrite output is used for hybrid_search (both dense and sparse).
A future improvement: use original query for sparse, rewritten for dense only —
BM25 keyword matching works better with the original terse query.
"""

from ingestion.chunkers import Document
from query.rewriter import rewrite
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank_async
from monitoring.tracer import span
from query.router import route


async def pipeline(query: str, allowed_repos: list[str] | None = None, trace_id: str = None) -> list[Document]:
    """
    Run the full retrieval pipeline and return top-5 most relevant chunks.

    Args:
        query: the raw user question

    Returns:
        top-5 Document chunks after rewriting, hybrid retrieval, and reranking
    """
    # Step 1: Route — classify query type to filter retrieval by content_type
    with span(trace_id, "route") as s:
        content_type = await route(query)
        s["metadata"]["content_type"] = content_type

    # Step 2: HyDE — generate a fake answer to use as the retrieval query
    # The fake answer uses FastAPI vocabulary, so it embeds closer to real chunks
    with span(trace_id, "rewrite") as s:
        hyde_response = await rewrite(query=query)
        s["metadata"]["query"] = query
        s["metadata"]["hyde_response"] = hyde_response[:200]

    # Step 3: Hybrid search — dense + BM25 + RRF fusion → top 20 candidates
    with span(trace_id, "hybrid_search") as s:
        response = await hybrid_search(query=hyde_response, allowed_repos=allowed_repos, content_type=content_type)
        s["metadata"]["num_results"] = len(response)

    # Step 3: Rerank — cross-encoder scores all 20 (query, chunk) pairs jointly → top 5
    # Uses original query here so reranker judges relevance to what user actually asked
    with span(trace_id, "rerank_async") as s:
        result = await rerank_async(query=query, docs=response)
        s["metadata"]["num_results"] = len(result)

    return result