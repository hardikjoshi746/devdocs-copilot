"""
retrieval/sparse.py

Sparse (keyword) retrieval using BM25.
Scores chunks by exact token matches — no vectors, no network calls.

Why sparse retrieval?
- Catches exact identifiers: HTTPException, Depends, @router.get
- These are rare tokens that BM25 weights heavily (high IDF)
- Dense search misses them because embeddings blur exact token identity
- Misses paraphrases — dense retrieval covers that
"""

from ingestion.chunkers import Document
from rank_bm25 import BM25Okapi
import json
from pathlib import Path


def sparse_search(query: str, n_results: int = 10, allowed_repos: list[str] | None = None) -> list[Document]:
    """
    Score all chunks against the query using BM25 and return the top-n.

    allowed_repos: if provided, only chunks whose metadata["repo"] is in this
    list are returned. Passes through Chroma's same repo-scoping guarantee for
    the sparse path.

    No async needed — BM25 is pure CPU math, no network calls.
    Loads the tokenized corpus from JSON (safe) and rebuilds BM25Okapi
    (deterministic — same corpus always produces the same index).
    """
    # Load tokenized corpus from JSON — safe alternative to binary serialization.
    # Deserializing an untrusted pickle file executes arbitrary Python (RCE vector).
    # JSON has no execution surface; BM25Okapi is rebuilt from raw token lists.
    with Path("data/bm25_corpus.json").open() as f:
        corpus = json.load(f)
    bm25 = BM25Okapi(corpus)

    # Load all chunks from JSONL — needed to map BM25 result indices back to Documents
    # BM25 works on positional indices, not ids, so we need the original ordered list
    chunks = []
    with Path("data/chunks/chunks.jsonl").open() as f:
        for line in f:
            chunks.append(Document(**json.loads(line)))

    # Tokenize query the same way as ingestion: lowercase + whitespace split
    # Consistency matters — if ingestion used "HTTPException" and query uses
    # "httpexception", they won't match. Both lowercase, so they do.
    tokenized = query.lower().split()

    # get_scores returns one float per chunk — higher = more relevant
    # BM25 weighs term frequency (how often token appears) and
    # inverse document frequency (how rare the token is across all chunks)
    scores = bm25.get_scores(tokenized)

    # Apply repo filter before ranking so restricted chunks never surface.
    # BM25 indices are positional — build a filtered view first.
    if allowed_repos:
        filtered = [
            (i, chunk) for i, chunk in enumerate(chunks)
            if chunk.metadata.get("repo") in allowed_repos
        ]
    else:
        filtered = list(enumerate(chunks))

    if not filtered:
        return []

    original_indices = [i for i, _ in filtered]
    filtered_chunks = [chunk for _, chunk in filtered]
    filtered_scores = [scores[i] for i in original_indices]

    top_indices = sorted(
        range(len(filtered_scores)),
        key=lambda i: filtered_scores[i],
        reverse=True,
    )[:n_results]

    return [filtered_chunks[i] for i in top_indices]