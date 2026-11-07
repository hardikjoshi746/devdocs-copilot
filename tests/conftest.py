"""
tests/conftest.py

Shared pytest configuration and fixtures.
"""

import os
import time
import pytest
import jwt
from unittest.mock import patch, AsyncMock
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# JWT test secret — set before any app module is imported so verify_token
# can find it. Must be ≥32 bytes to avoid PyJWT InsecureKeyLengthWarning.
# ---------------------------------------------------------------------------
TEST_JWT_SECRET = "test-secret-key-must-be-32bytes-long!!!!"
os.environ["JWT_SIGNING_SECRET"] = TEST_JWT_SECRET


@pytest.fixture
def valid_token():
    """JWT signed with TEST_JWT_SECRET, repos=["fastapi"], 1-hour expiry."""
    return jwt.encode(
        {"sub": "test_tenant", "repos": ["fastapi"], "exp": int(time.time()) + 3600},
        TEST_JWT_SECRET,
        algorithm="HS256",
    )


@pytest.fixture
def expired_token():
    """JWT that expired 1 second ago."""
    return jwt.encode(
        {"sub": "test_tenant", "repos": ["fastapi"], "exp": int(time.time()) - 1},
        TEST_JWT_SECRET,
        algorithm="HS256",
    )


@pytest.fixture
def wrong_secret_token():
    """JWT signed with a different secret — signature won't verify."""
    return jwt.encode(
        {"sub": "test_tenant", "repos": ["fastapi"], "exp": int(time.time()) + 3600},
        "completely-different-secret-key-32b!!",
        algorithm="HS256",
    )


@pytest.fixture
def auth_client(valid_token):
    """
    FastAPI TestClient with JWT dependency overridden.

    Use this in business-logic tests (pipeline, GOOD/ABSTAIN/EXPAND paths)
    so they don't have to care about auth. Auth itself is tested in
    test_security.py using the raw TestClient.

    Also patches _redis=None so tests don't need a running Redis instance.
    """
    from api.main import app, verify_token
    app.dependency_overrides[verify_token] = lambda: ("test_tenant", ["fastapi"])
    # Mock the async LLM classifier so tests don't make real API calls
    # and don't depend on ANTHROPIC_API_KEY being set.
    with patch("api.main._redis", None), \
         patch("api.guardrails._llm_injection_check", new=AsyncMock(return_value=False)):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        yield client
    app.dependency_overrides.clear()