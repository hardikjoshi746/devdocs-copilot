"""
retrieval/reranker.py

Cross-encoder reranking — takes the top-20 hybrid results and re-scores
them with a model that reads query and chunk jointly.

Why rerank?
- Bi-encoders (dense search) embed query and chunk separately — fast but less accurate
- Cross-encoders read both together — slower but far more accurate
- We can't use cross-encoders on all 2500+ chunks (~4min per query)
- Solution: hybrid search narrows to top-20, cross-encoder re-scores just those 20 (~2s)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Free, runs locally on CPU — no API key needed
- Trained on MS MARCO passage ranking dataset
- ~80MB, downloaded once and cached by HuggingFace
"""

from sentence_transformers import CrossEncoder
from ingestion.chunkers import Document


def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    """
    Re-score docs against the query using a cross-encoder and return top-n.

    No async needed — runs locally on CPU, no network calls after first download.

    Args:
        query: the user's question
        docs:  candidate chunks from hybrid_search (typically top-20)
        top_n: how many to return after reranking (typically 5)
    """
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Build (query, chunk) pairs — the cross-encoder reads both together
    # to judge relevance more accurately than cosine similarity alone
    pairs = [[query, doc.content] for doc in docs]

    # predict() scores all pairs in one batch — faster than calling one by one
    # returns a numpy array of floats, one score per pair
    scores = model.predict(pairs)

    # zip pairs each score with its Document, sort by score descending,
    # then unpack just the Documents
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]