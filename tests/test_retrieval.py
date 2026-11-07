"""
tests/test_retrieval.py

Integration tests for retrieval layer — sparse, dense, hybrid, reranker.
Requires data/bm25.pkl, data/chunks/chunks.jsonl, and data/chroma/ to exist.
Run ingestion first: python -m ingestion.run_ingestion
"""

import pytest
import asyncio
from ingestion.chunkers import Document
from retrieval.sparse import sparse_search
from retrieval.dense import dense_search
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(id="test::func", content="def hello(): pass", type="code"):
    return Document(
        id=id, content=content, type=type,
        source="test.py", parent_id=None, metadata={}
    )


# ---------------------------------------------------------------------------
# sparse_search
# ---------------------------------------------------------------------------

def test_sparse_returns_list():
    results = sparse_search("HTTPException")
    assert isinstance(results, list)


def test_sparse_returns_documents():
    results = sparse_search("HTTPException")
    assert all(isinstance(d, Document) for d in results)


def test_sparse_respects_n_results():
    results = sparse_search("HTTPException", n_results=3)
    assert len(results) <= 3


def test_sparse_relevant_query():
    results = sparse_search("HTTPException", n_results=5)
    ids = [r.id for r in results]
    # At least one result should be from handling-errors or exceptions
    assert any("exception" in id.lower() or "error" in id.lower() for id in ids)


def test_sparse_empty_query_returns_results():
    # BM25 with empty query returns all zeros — still returns n_results docs
    results = sparse_search("", n_results=5)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# dense_search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dense_returns_list():
    results = await dense_search("HTTPException error handling")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_dense_returns_documents():
    results = await dense_search("HTTPException error handling")
    assert all(isinstance(d, Document) for d in results)


@pytest.mark.asyncio
async def test_dense_respects_n_results():
    results = await dense_search("dependency injection", n_results=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_dense_relevant_query():
    results = await dense_search("how to handle HTTP errors in FastAPI", n_results=5)
    ids = " ".join(r.id for r in results)
    assert "error" in ids.lower() or "exception" in ids.lower()


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_returns_list():
    results = await hybrid_search("HTTPException")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_returns_documents():
    results = await hybrid_search("HTTPException")
    assert all(isinstance(d, Document) for d in results)


@pytest.mark.asyncio
async def test_hybrid_respects_n_results():
    results = await hybrid_search("dependency injection", n_results=5)
    assert len(results) <= 5


@pytest.mark.asyncio
async def test_hybrid_unique_ids():
    results = await hybrid_search("HTTPException", n_results=10)
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids))  # no duplicates from merging dense + sparse


@pytest.mark.asyncio
async def test_hybrid_relevant_query():
    results = await hybrid_search("how does HTTPException work", n_results=5)
    ids = " ".join(r.id for r in results)
    assert "exception" in ids.lower() or "error" in ids.lower()


# ---------------------------------------------------------------------------
# reranker
# ---------------------------------------------------------------------------

def test_rerank_returns_list():
    docs = [make_doc(id=f"test::{i}", content=f"content {i}") for i in range(5)]
    result = rerank("hello function", docs)
    assert isinstance(result, list)


def test_rerank_returns_documents():
    docs = [make_doc(id=f"test::{i}", content=f"content {i}") for i in range(5)]
    result = rerank("hello function", docs)
    assert all(isinstance(d, Document) for d in result)


def test_rerank_respects_top_n():
    docs = [make_doc(id=f"test::{i}", content=f"content {i}") for i in range(10)]
    result = rerank("hello", docs, top_n=3)
    assert len(result) == 3


def test_rerank_fewer_docs_than_top_n():
    docs = [make_doc(id=f"test::{i}") for i in range(2)]
    result = rerank("hello", docs, top_n=5)
    assert len(result) == 2


def test_rerank_most_relevant_first():
    docs = [
        make_doc(id="irrelevant", content="the weather today is sunny"),
        make_doc(id="relevant", content="HTTPException is raised with status_code=404"),
    ]
    result = rerank("how does HTTPException work", docs, top_n=2)
    assert result[0].id == "relevant"


def test_rerank_empty_docs():
    result = rerank("hello", [], top_n=5)
    assert result == []