"""
retrieval/hybrid.py

Hybrid retrieval: fuses dense (Chroma) and sparse (BM25) results using
Reciprocal Rank Fusion (RRF).

Why hybrid?
- Dense alone misses exact identifiers (HTTPException, Depends)
- Sparse alone misses paraphrases ("handle errors" vs "raise HTTPException")
- RRF combines both without needing compatible score scales

Why RRF over score averaging?
- Dense scores are cosine similarities (0-1)
- BM25 scores are raw term frequencies (0-100+)
- These scales are incompatible — you can't average them
- RRF ignores raw scores entirely and uses rank position instead
- Formula: score(chunk) = Σ 1 / (k + rank),  k=60
- k=60 dampens rank-1 dominance — empirically optimal across many tasks
"""

from ingestion.chunkers import Document
from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from collections import defaultdict


async def hybrid_search(query: str, n_results: int = 10) -> list[Document]:
    """
    Run dense and sparse search, fuse results with RRF, return top-n.

    Fetches 20 results from each retriever so RRF has enough candidates
    to work with — chunks that rank well in both lists float to the top.
    """
    # Fetch top-20 from each retriever independently
    # Dense is async (OpenAI API call), sparse is sync (pure CPU)
    dense_results = await dense_search(query, n_results=20)
    sparse_results = sparse_search(query, n_results=20)

    # RRF accumulator — defaultdict(float) initializes missing keys to 0.0
    # so we can += without checking if the key exists first
    scores: dict[str, float] = defaultdict(float)

    # doc_map lets us retrieve the Document object by id after scoring
    doc_map: dict[str, Document] = {}

    # enumerate(start=1) gives rank 1, 2, 3... (not 0-indexed)
    # rank 1 → 1/(60+1) = 0.0164, rank 2 → 1/(60+2) = 0.0161, etc.
    for rank, doc in enumerate(dense_results, start=1):
        scores[doc.id] += 1 / (60 + rank)
        doc_map[doc.id] = doc

    for rank, doc in enumerate(sparse_results, start=1):
        scores[doc.id] += 1 / (60 + rank)
        doc_map[doc.id] = doc

    # Sort all seen chunk ids by their accumulated RRF score, descending
    sorted_ids = sorted(scores, key=lambda id: scores[id], reverse=True)[:n_results]

    return [doc_map[id] for id in sorted_ids]