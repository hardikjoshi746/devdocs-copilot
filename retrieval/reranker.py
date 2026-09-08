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

import asyncio
from functools import partial
from sentence_transformers import CrossEncoder
from ingestion.chunkers import Document

# Model is lazy-loaded on first call — ~80MB, takes a few seconds
# The import above is at module level (main thread) to avoid thread-safety issues
_model = None

def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def _rerank_sync(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Synchronous reranking — runs in a thread pool to avoid blocking event loop."""
    if not docs:
        return []
    model = _get_model()
    # Truncate content to 2000 chars — cross-encoder tokenizer caps at 512 tokens anyway
    pairs = [[query, doc.content[:2000]] for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def rerank(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    """
    Re-score docs against the query using a cross-encoder and return top-n.

    Runs synchronously — call from sync code or use rerank_async from async code.
    """
    return _rerank_sync(query, docs, top_n)


async def rerank_async(query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
    """
    Async wrapper — runs the CPU-heavy cross-encoder in a thread pool
    so it doesn't block the event loop.

    Use this from async contexts (pipeline, run_ablations).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_rerank_sync, query, docs, top_n))