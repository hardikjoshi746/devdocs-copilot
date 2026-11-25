"""
tests/test_mcp.py

Unit tests for mcp__server.py — the MCP tool that proxies /query.
No real HTTP calls are made; httpx is patched.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, json_body: dict) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = json_body
    return mock


GOOD_RESPONSE = {
    "answer": "The scraper uses httpx to fetch job listings.",
    "citations": [{"source": "scraper.py", "chunk_id": "job_scrapper::scraper.py::fetch"}],
    "retrieval_quality": "GOOD",
}

EXPAND_RESPONSE = {
    "answer": "Job scraping works by querying multiple sources.",
    "citations": [{"source": "main.py", "chunk_id": "job_scrapper::main.py::run"}],
    "retrieval_quality": "EXPAND",
}


# ---------------------------------------------------------------------------
# query_codebase — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_codebase_returns_answer():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(200, GOOD_RESPONSE))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client):
        result = await query_codebase("how does the scraper work?")

    assert "The scraper uses httpx" in result


@pytest.mark.asyncio
async def test_query_codebase_includes_citations():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(200, GOOD_RESPONSE))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client):
        result = await query_codebase("how does the scraper work?")

    assert "job_scrapper::scraper.py::fetch" in result


@pytest.mark.asyncio
async def test_query_codebase_includes_quality_badge():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(200, GOOD_RESPONSE))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client):
        result = await query_codebase("how does the scraper work?")

    assert "GOOD" in result


@pytest.mark.asyncio
async def test_query_codebase_expand_quality_shown():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(200, EXPAND_RESPONSE))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client):
        result = await query_codebase("explain job scraping")

    assert "EXPAND" in result


# ---------------------------------------------------------------------------
# 503 — retrieval quality too low
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_codebase_503_returns_friendly_message():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(503, {"error": "retrieval_quality_too_low"}))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client):
        result = await query_codebase("asdflkjh gibberish")

    assert "retrieval quality too low" in result.lower()


# ---------------------------------------------------------------------------
# Connection error — API not running
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_codebase_connect_error_returns_friendly_message():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client):
        result = await query_codebase("anything")

    assert "not running" in result.lower() or "uvicorn" in result.lower()


# ---------------------------------------------------------------------------
# Auth header — token is included when DEVDOCS_TOKEN is set
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_codebase_sends_auth_header_when_token_set():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(200, GOOD_RESPONSE))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client), \
         patch("mcp__server.DEVDOCS_TOKEN", "test-token-123"):
        await query_codebase("test question")

    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["headers"].get("Authorization") == "Bearer test-token-123"


@pytest.mark.asyncio
async def test_query_codebase_no_auth_header_when_token_empty():
    from mcp__server import query_codebase

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=_mock_response(200, GOOD_RESPONSE))

    with patch("mcp__server.httpx.AsyncClient", return_value=mock_client), \
         patch("mcp__server.DEVDOCS_TOKEN", ""):
        await query_codebase("test question")

    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["headers"] == {}