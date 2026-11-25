"""
tests/test_router.py

Unit tests for query/router.py.
Mocks the Anthropic client so no real API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _mock_haiku(response_text: str):
    """Return a patched AsyncAnthropic whose messages.create returns response_text."""
    mock_client = AsyncMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=response_text)]
    )
    return mock_client


# ---------------------------------------------------------------------------
# Valid categories
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_returns_code():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("code")):
        assert await route("how does the scraper function work?") == "code"


@pytest.mark.asyncio
async def test_route_returns_doc():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("doc")):
        assert await route("what does the README say about setup?") == "doc"


@pytest.mark.asyncio
async def test_route_returns_issue():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("issue")):
        assert await route("is there a bug with authentication?") == "issue"


@pytest.mark.asyncio
async def test_route_returns_general():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("general")):
        assert await route("explain the overall architecture") == "general"


# ---------------------------------------------------------------------------
# Normalisation — uppercase / mixed case must be lowercased
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_normalises_uppercase():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("CODE")):
        assert await route("anything") == "code"


@pytest.mark.asyncio
async def test_route_normalises_mixed_case():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("Doc")):
        assert await route("anything") == "doc"


# ---------------------------------------------------------------------------
# Fallback — unknown response defaults to "general"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_unknown_response_falls_back_to_general():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("I think this is code")):
        assert await route("anything") == "general"


@pytest.mark.asyncio
async def test_route_empty_response_falls_back_to_general():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("")):
        assert await route("anything") == "general"


@pytest.mark.asyncio
async def test_route_whitespace_response_falls_back_to_general():
    from query.router import route
    with patch("query.router.AsyncAnthropic", return_value=_mock_haiku("   ")):
        assert await route("anything") == "general"