"""Tests for FraudAI JWT authentication.

Covers:
- Dev mode accepts ``dev-token``
- Valid JWT returns correct user
- Expired JWT returns 401
- Invalid JWT returns 401
- ``create_token`` generates valid tokens
- ``/auth/token`` login endpoint returns a usable token
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import jwt
import pytest
from fastapi import FastAPI, HTTPException

from fraudai.api.auth import (
    JWT_ALGORITHM,
    JWT_AUDIENCE,
    JWT_EXPIRATION_HOURS,
    JWT_ISSUER,
    create_token,
    get_current_user,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Dev-only signing key — matches the default in Settings.
_DEV_SIGNING_KEY = "fraudai-dev-" + "secret-change-in-production"


def _mock_settings(**overrides: Any) -> MagicMock:
    """Build a mock settings object for auth tests."""
    defaults: dict[str, Any] = {
        "environment": "development",
        "jwt_secret": _DEV_SIGNING_KEY,
        "ollama_host": "http://localhost:11434",
        "llm_provider": "groq",
        "anthropic_api_key": "",
        "groq_api_key": "",
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _forge_token(
    *, drop: tuple[str, ...] = (), signing_key: str = _DEV_SIGNING_KEY, **claims: Any
) -> str:
    """Sign a token with valid default claims, overriding or dropping some."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": "alice",
        "tenant_id": "acme",
        "tier": "pro",
        "exp": now + timedelta(hours=1),
        "iat": now,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
    }
    payload.update(claims)
    for name in drop:
        payload.pop(name, None)
    return jwt.encode(payload, signing_key, algorithm=JWT_ALGORITHM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app() -> FastAPI:
    """Minimal FastAPI app with routes for auth testing (no dependency overrides)."""
    from fraudai.api.routes import router

    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")

    # Minimal app state needed for routes that we do NOT test here
    test_app.state.graph = MagicMock()
    test_app.state.store = MagicMock()

    from fraudai.api.session_manager import SessionManager

    test_app.state.session_manager = SessionManager()
    test_app.state.feedback_store = []
    test_app.state.embedder = None

    return test_app


@pytest.fixture()
async def client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# create_token
# ---------------------------------------------------------------------------


class TestCreateToken:
    """Tests for the ``create_token`` helper."""

    def test_returns_string(self) -> None:
        with patch("fraudai.core.config.settings", _mock_settings()):
            token = create_token(user_id="alice", tenant_id="t1", tier="pro")
        assert isinstance(token, str)

    def test_payload_contains_claims(self) -> None:
        with patch("fraudai.core.config.settings", _mock_settings()):
            token = create_token(
                user_id="bob",
                tenant_id="acme",
                tier="enterprise",
                email="bob@acme.com",
                is_admin=True,
            )
        payload = jwt.decode(
            token, _DEV_SIGNING_KEY, algorithms=[JWT_ALGORITHM], audience=JWT_AUDIENCE
        )
        assert payload["sub"] == "bob"
        assert payload["tenant_id"] == "acme"
        assert payload["tier"] == "enterprise"
        assert payload["email"] == "bob@acme.com"
        assert payload["is_admin"] is True
        assert payload["iss"] == JWT_ISSUER
        assert payload["aud"] == JWT_AUDIENCE

    def test_token_has_expiration(self) -> None:
        with patch("fraudai.core.config.settings", _mock_settings()):
            token = create_token(user_id="carol", tenant_id="t1", tier="free")
        payload = jwt.decode(
            token, _DEV_SIGNING_KEY, algorithms=[JWT_ALGORITHM], audience=JWT_AUDIENCE
        )
        assert "exp" in payload
        assert "iat" in payload
        # exp should be ~24 hours from now
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=UTC)
        assert exp_dt > datetime.now(UTC)
        assert exp_dt < datetime.now(UTC) + timedelta(hours=JWT_EXPIRATION_HOURS + 1)

    def test_token_decodable_with_correct_key(self) -> None:
        with patch("fraudai.core.config.settings", _mock_settings()):
            token = create_token(user_id="dave", tenant_id="t1", tier="pro")
        # Should not raise
        jwt.decode(token, _DEV_SIGNING_KEY, algorithms=[JWT_ALGORITHM], audience=JWT_AUDIENCE)

    def test_token_fails_with_wrong_key(self) -> None:
        with patch("fraudai.core.config.settings", _mock_settings()):
            token = create_token(user_id="eve", tenant_id="t1", tier="free")
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(token, "wrong-key-value", algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# get_current_user -- unit tests
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    """Tests for the ``get_current_user`` dependency."""

    async def test_dev_token_returns_dev_user(self) -> None:
        """``dev-token`` in development mode returns the hardcoded dev user."""
        with patch("fraudai.core.config.settings", _mock_settings(environment="development")):
            user = await get_current_user(token="dev-token")
        assert user.user_id == "dev-user"
        assert user.tenant_id == "dev-tenant"
        assert user.tier == "enterprise"
        assert user.is_admin is True

    async def test_dev_token_rejected_in_staging(self) -> None:
        """``dev-token`` must not grant admin outside development."""
        with (
            patch("fraudai.core.config.settings", _mock_settings(environment="staging")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_current_user(token="dev-token")
        assert exc_info.value.status_code == 401

    async def test_valid_jwt_returns_user(self) -> None:
        """A properly signed JWT returns the corresponding User."""
        mock_s = _mock_settings(environment="production")
        with patch("fraudai.core.config.settings", mock_s):
            token = create_token(
                user_id="alice",
                tenant_id="acme",
                tier="pro",
                email="alice@acme.com",
                is_admin=False,
            )
            user = await get_current_user(token=token)

        assert user.user_id == "alice"
        assert user.tenant_id == "acme"
        assert user.tier == "pro"
        assert user.email == "alice@acme.com"
        assert user.is_admin is False

    async def test_valid_jwt_in_dev_mode(self) -> None:
        """A real JWT also works in dev mode (not just dev-token)."""
        mock_s = _mock_settings(environment="development")
        with patch("fraudai.core.config.settings", mock_s):
            token = create_token(user_id="bob", tenant_id="t1", tier="enterprise")
            user = await get_current_user(token=token)
        assert user.user_id == "bob"

    async def test_expired_jwt_returns_401(self) -> None:
        """An expired JWT raises HTTPException 401 with 'Token expired'."""
        token = _forge_token(
            sub="expired-user",
            exp=datetime.now(UTC) - timedelta(hours=1),
            iat=datetime.now(UTC) - timedelta(hours=25),
        )

        mock_s = _mock_settings(environment="production")
        with patch("fraudai.core.config.settings", mock_s):
            with pytest.raises(Exception) as exc_info:
                await get_current_user(token=token)
            assert exc_info.value.status_code == 401
            assert "Token expired" in str(exc_info.value.detail)

    async def test_invalid_jwt_returns_401(self) -> None:
        """A malformed or wrongly-signed JWT raises HTTPException 401."""
        mock_s = _mock_settings(environment="production")
        with patch("fraudai.core.config.settings", mock_s):
            with pytest.raises(Exception) as exc_info:
                await get_current_user(token="not-a-valid-jwt")
            assert exc_info.value.status_code == 401
            assert "Invalid token" in str(exc_info.value.detail)

    async def test_wrong_signing_key_returns_401(self) -> None:
        """A JWT signed with a different key is rejected."""
        token = jwt.encode(
            {
                "sub": "user",
                "tenant_id": "t1",
                "tier": "free",
                "exp": datetime.now(UTC) + timedelta(hours=1),
                "iat": datetime.now(UTC),
            },
            "different-signing-key",
            algorithm=JWT_ALGORITHM,
        )
        mock_s = _mock_settings(environment="production")
        with patch("fraudai.core.config.settings", mock_s):
            with pytest.raises(Exception) as exc_info:
                await get_current_user(token=token)
            assert exc_info.value.status_code == 401

    async def test_dev_token_rejected_in_production(self) -> None:
        """``dev-token`` is NOT accepted in production mode."""
        mock_s = _mock_settings(environment="production")
        with patch("fraudai.core.config.settings", mock_s):
            with pytest.raises(Exception) as exc_info:
                await get_current_user(token="dev-token")
            assert exc_info.value.status_code == 401


class TestMalformedTokens:
    """Any structurally bad token is a 401, never an unhandled 500 (B4)."""

    @pytest.mark.parametrize(
        "token",
        [
            pytest.param(_forge_token(drop=("sub",)), id="missing_sub"),
            pytest.param(_forge_token(sub=""), id="empty_sub"),
            pytest.param(_forge_token(sub=42), id="non_string_sub"),
            pytest.param(_forge_token(drop=("tenant_id",)), id="missing_tenant"),
            pytest.param(_forge_token(tenant_id=""), id="empty_tenant"),
            pytest.param(_forge_token(tenant_id=["a"]), id="non_string_tenant"),
            pytest.param(_forge_token(drop=("tier",)), id="missing_tier"),
            pytest.param(_forge_token(tier="root"), id="invalid_tier"),
            pytest.param(_forge_token(tier=1), id="non_string_tier"),
            pytest.param(_forge_token(is_admin="yes"), id="string_is_admin"),
            pytest.param(_forge_token(is_admin=1), id="int_is_admin"),
            pytest.param(_forge_token(email=["x"]), id="non_string_email"),
            pytest.param(_forge_token(drop=("exp",)), id="missing_exp"),
            pytest.param(_forge_token(drop=("iss",)), id="missing_iss"),
            pytest.param(_forge_token(drop=("aud",)), id="missing_aud"),
            pytest.param(_forge_token(iss="someone-else"), id="wrong_issuer"),
            pytest.param(_forge_token(aud="another-api"), id="wrong_audience"),
            pytest.param(_forge_token(signing_key="x" * 40), id="wrong_signature"),
            pytest.param("", id="empty_token"),
            pytest.param("a.b.c", id="garbage_segments"),
        ],
    )
    async def test_rejected_with_401(self, token: str) -> None:
        with (
            patch("fraudai.core.config.settings", _mock_settings(environment="production")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_current_user(token=token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    async def test_alg_none_token_is_rejected(self) -> None:
        unsigned = jwt.encode(
            {
                "sub": "a",
                "tenant_id": "t",
                "tier": "free",
                "iss": JWT_ISSUER,
                "aud": JWT_AUDIENCE,
                "exp": datetime.now(UTC) + timedelta(hours=1),
                "iat": datetime.now(UTC),
            },
            key=None,
            algorithm="none",
        )
        with (
            patch("fraudai.core.config.settings", _mock_settings(environment="production")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_current_user(token=unsigned)
        assert exc_info.value.status_code == 401

    async def test_malformed_token_over_http_is_401_not_500(
        self, client: httpx.AsyncClient
    ) -> None:
        token = _forge_token(drop=("sub",))
        with patch("fraudai.core.config.settings", _mock_settings(environment="production")):
            resp = await client.get(
                f"/api/v1/sessions/{'0' * 8}-0000-4000-8000-{'0' * 12}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 401

    async def test_optional_claims_default_safely(self) -> None:
        token = _forge_token()
        with patch("fraudai.core.config.settings", _mock_settings(environment="production")):
            user = await get_current_user(token=token)
        assert user.email == ""
        assert user.is_admin is False


# ---------------------------------------------------------------------------
# POST /auth/token -- integration tests
# ---------------------------------------------------------------------------


class TestLoginEndpoint:
    """Tests for the ``/api/v1/auth/token`` endpoint."""

    async def test_login_returns_token_in_dev(self, client: httpx.AsyncClient) -> None:
        """Login in dev mode returns a valid JWT."""
        with patch("fraudai.api.routes.settings", _mock_settings(environment="development")):
            resp = await client.post(
                "/api/v1/auth/token",
                data={"username": "testuser", "password": "testpass"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == JWT_EXPIRATION_HOURS * 3600

        # The returned token should be a decodable JWT
        payload = jwt.decode(
            data["access_token"],
            _DEV_SIGNING_KEY,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
        )
        assert payload["sub"] == "testuser"
        assert payload["tenant_id"] == "default"
        assert payload["tier"] == "enterprise"

    async def test_login_returns_501_in_staging(self, client: httpx.AsyncClient) -> None:
        """Staging has no free login: any credentials must not mint a token."""
        with patch("fraudai.api.routes.settings", _mock_settings(environment="staging")):
            resp = await client.post(
                "/api/v1/auth/token",
                data={"username": "anyone", "password": "anything"},
            )
        assert resp.status_code == 501
        assert "access_token" not in resp.json()

    async def test_login_returns_501_in_production(self, client: httpx.AsyncClient) -> None:
        """Login in production returns 501 (not yet implemented)."""
        with patch("fraudai.api.routes.settings", _mock_settings(environment="production")):
            resp = await client.post(
                "/api/v1/auth/token",
                data={"username": "user", "password": "pass"},
            )
        assert resp.status_code == 501
        assert "Production authentication not configured" in resp.json()["detail"]

    async def test_login_token_works_for_auth(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        """A token obtained from /auth/token can authenticate API calls."""
        mock_s = _mock_settings(environment="development")

        # Step 1: Obtain token
        with patch("fraudai.api.routes.settings", mock_s):
            resp = await client.post(
                "/api/v1/auth/token",
                data={"username": "integration-user", "password": "pass"},
            )
        assert resp.status_code == 200
        token = resp.json()["access_token"]

        # Step 2: Use token to call an authenticated endpoint (GET /sessions/<id>)
        # We expect a 404 because the session doesn't exist, NOT a 401
        with patch("fraudai.core.config.settings", mock_s):
            resp = await client.get(
                "/api/v1/sessions/nonexistent",
                headers={"Authorization": f"Bearer {token}"},
            )
        # 404 means auth succeeded but session wasn't found -- correct behaviour
        assert resp.status_code == 404

    async def test_no_token_returns_401(self, client: httpx.AsyncClient) -> None:
        """Calling an authenticated endpoint without a token returns 401."""
        resp = await client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert resp.status_code == 401
