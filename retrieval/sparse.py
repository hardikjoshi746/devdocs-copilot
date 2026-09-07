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
import pickle
import json
from pathlib import Path


def sparse_search(query: str, n_results: int = 10) -> list[Document]:
    """
    Score all chunks against the query using BM25 and return the top-n.

    No async needed — BM25 is pure CPU math, no network calls.
    Loads the BM25 index and chunk list from disk on every call.
    (In production this would be cached in memory at startup.)
    """
    # Load BM25 index from pickle — "rb" = read binary (pickle is binary format)
    # Built by embed_and_store.py using the same tokenization used here
    with Path("data/bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)

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

    # Sort indices by score descending, take top n_results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [chunks[i] for i in top_indices]