"""
tests/test_content_type_filter.py

Tests for content_type filtering added in Session 5 (query router).
Covers dense_search (Chroma where-filter) and sparse_search (post-filter).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ingestion.chunkers import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(id, content, doc_type, repo="testrepo"):
    return Document(
        id=id, content=content, type=doc_type,
        source="test.py", parent_id=None, metadata={"repo": repo},
    )


def _sparse_path_factory(docs):
    """Build Path mock for sparse_search with given docs as the corpus."""
    from dataclasses import asdict
    corpus = [d.content.lower().split() for d in docs]
    jsonl_lines = [json.dumps(asdict(d)) for d in docs]

    class _FakeCM:
        def __init__(self, lines):
            self._lines = lines
        def __enter__(self):
            return iter(self._lines)
        def __exit__(self, *a):
            return False

    def factory(path_str):
        m = MagicMock()
        if "corpus" in str(path_str):
            m.open.return_value = _FakeCM([""])
        else:
            m.open.return_value = _FakeCM(jsonl_lines)
        return m

    return corpus, factory


# ===========================================================================
# dense_search — content_type filter builds correct Chroma where clause
# ===========================================================================

@pytest.mark.asyncio
async def test_dense_content_type_builds_eq_filter():
    """content_type="code" → where={"type": {"$eq": "code"}}"""
    from retrieval.dense import dense_search

    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )

    with patch("retrieval.dense._openai_client", mock_openai), \
         patch("retrieval.dense.chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        await dense_search("test", content_type="code")

    where = mock_collection.query.call_args[1].get("where")
    assert where == {"type": {"$eq": "code"}}


@pytest.mark.asyncio
async def test_dense_content_type_and_repo_builds_and_filter():
    """Both filters → where={"$and": [...]}"""
    from retrieval.dense import dense_search

    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )

    with patch("retrieval.dense._openai_client", mock_openai), \
         patch("retrieval.dense.chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        await dense_search("test", allowed_repos=["job_scrapper"], content_type="doc")

    where = mock_collection.query.call_args[1].get("where")
    assert where == {"$and": [
        {"repo": {"$in": ["job_scrapper"]}},
        {"type": {"$eq": "doc"}},
    ]}


@pytest.mark.asyncio
async def test_dense_no_content_type_no_filter():
    """content_type=None → no type filter added."""
    from retrieval.dense import dense_search

    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )

    with patch("retrieval.dense._openai_client", mock_openai), \
         patch("retrieval.dense.chromadb.PersistentClient") as mock_chroma:
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        await dense_search("test", content_type=None)

    where = mock_collection.query.call_args[1].get("where")
    assert where is None


# ===========================================================================
# sparse_search — content_type post-filter
# ===========================================================================

def test_sparse_content_type_filters_to_code():
    """content_type="code" → only code chunks returned."""
    from retrieval.sparse import sparse_search

    docs = [
        _make_doc("r::code_func", "def scrape(): pass", "code"),
        _make_doc("r::doc_section", "# Setup\ninstall the package", "doc"),
        _make_doc("issue::1", "bug: login fails", "issue"),
    ]
    corpus, path_factory = _sparse_path_factory(docs)
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [1.0, 1.0, 1.0]

    with patch("retrieval.sparse.BM25Okapi", return_value=mock_bm25), \
         patch("retrieval.sparse.json.load", return_value=corpus), \
         patch("retrieval.sparse.Path", side_effect=path_factory):
        results = sparse_search("scrape", content_type="code")

    types = {r.type for r in results}
    assert types == {"code"}


def test_sparse_content_type_filters_to_doc():
    from retrieval.sparse import sparse_search

    docs = [
        _make_doc("r::code_func", "def scrape(): pass", "code"),
        _make_doc("r::doc_section", "# Setup\ninstall the package", "doc"),
    ]
    corpus, path_factory = _sparse_path_factory(docs)
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [1.0, 1.0]

    with patch("retrieval.sparse.BM25Okapi", return_value=mock_bm25), \
         patch("retrieval.sparse.json.load", return_value=corpus), \
         patch("retrieval.sparse.Path", side_effect=path_factory):
        results = sparse_search("setup", content_type="doc")

    assert all(r.type == "doc" for r in results)


def test_sparse_content_type_none_returns_all_types():
    """content_type=None → no type filter, all chunk types returned."""
    from retrieval.sparse import sparse_search

    docs = [
        _make_doc("r::code_func", "def scrape(): pass", "code"),
        _make_doc("r::doc_section", "# Setup", "doc"),
        _make_doc("issue::1", "bug report", "issue"),
    ]
    corpus, path_factory = _sparse_path_factory(docs)
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [1.0, 1.0, 1.0]

    with patch("retrieval.sparse.BM25Okapi", return_value=mock_bm25), \
         patch("retrieval.sparse.json.load", return_value=corpus), \
         patch("retrieval.sparse.Path", side_effect=path_factory):
        results = sparse_search("test", content_type=None)

    types = {r.type for r in results}
    assert types == {"code", "doc", "issue"}


def test_sparse_content_type_and_repo_filter_combined():
    """Both repo and content_type filters must both be applied."""
    from retrieval.sparse import sparse_search

    docs = [
        _make_doc("repo_a::code", "def foo(): pass", "code", repo="repo_a"),
        _make_doc("repo_b::code", "def bar(): pass", "code", repo="repo_b"),
        _make_doc("repo_a::doc", "# README", "doc", repo="repo_a"),
    ]
    corpus, path_factory = _sparse_path_factory(docs)
    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [1.0, 1.0, 1.0]

    with patch("retrieval.sparse.BM25Okapi", return_value=mock_bm25), \
         patch("retrieval.sparse.json.load", return_value=corpus), \
         patch("retrieval.sparse.Path", side_effect=path_factory):
        results = sparse_search("foo", allowed_repos=["repo_a"], content_type="code")

    ids = [r.id for r in results]
    assert "repo_a::code" in ids
    assert "repo_b::code" not in ids
    assert "repo_a::doc" not in ids


# ===========================================================================
# hybrid_search — "general" content_type passes None to both retrievers
# ===========================================================================

@pytest.mark.asyncio
async def test_hybrid_general_passes_none_to_retrievers():
    """content_type="general" must be converted to None before passing down."""
    from retrieval.hybrid import hybrid_search

    mock_dense = AsyncMock(return_value=[])
    mock_sparse = AsyncMock(return_value=[])

    with patch("retrieval.hybrid.dense_search", mock_dense), \
         patch("retrieval.hybrid.sparse_search_async", mock_sparse):
        await hybrid_search("test", content_type="general")

    dense_kwargs = mock_dense.call_args[1]
    sparse_kwargs = mock_sparse.call_args[1]
    assert dense_kwargs.get("content_type") is None
    assert sparse_kwargs.get("content_type") is None


@pytest.mark.asyncio
async def test_hybrid_code_passes_code_to_retrievers():
    from retrieval.hybrid import hybrid_search

    mock_dense = AsyncMock(return_value=[])
    mock_sparse = AsyncMock(return_value=[])

    with patch("retrieval.hybrid.dense_search", mock_dense), \
         patch("retrieval.hybrid.sparse_search_async", mock_sparse):
        await hybrid_search("test", content_type="code")

    assert mock_dense.call_args[1].get("content_type") == "code"
    assert mock_sparse.call_args[1].get("content_type") == "code"