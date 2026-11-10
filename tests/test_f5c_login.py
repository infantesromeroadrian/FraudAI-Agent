"""F5c.0: minimal local login (D5/R7).

The user is configured on the server (``AUTH_*``); identity and tier come from that
configuration, never from the request. Every credential here is synthetic.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import jwt
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from fraudai.api import auth as auth_module
from fraudai.api.auth import (
    JWT_ALGORITHM,
    JWT_AUDIENCE,
    LoginThrottle,
    hash_password,
    verify_password,
)
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager
from fraudai.core.config import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

SYNTHETIC_PASSWORD = "synthetic-pass-for-tests"  # noqa: S105 - fixture, not a credential
SYNTHETIC_SECRET = "s" * 48  # noqa: S105 - fixture signing key
FAST_COST = {"n": 16, "r": 8, "p": 1}  # cheap scrypt for tests; production uses the default


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "production",
        "jwt_secret": SYNTHETIC_SECRET,
        "auth_username": "ana",
        "auth_password_hash": hash_password(SYNTHETIC_PASSWORD, **FAST_COST),
        "auth_tenant_id": "tenant-ana",
        "auth_tier": "pro",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@contextmanager
def _using(cfg: Settings) -> Iterator[None]:
    with (
        patch("fraudai.api.routes.settings", cfg),
        patch("fraudai.core.config.settings", cfg),
    ):
        yield


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch: pytest.MonkeyPatch) -> LoginThrottle:
    throttle = LoginThrottle(max_failures=5, lockout_seconds=60)
    monkeypatch.setattr(auth_module, "login_throttle", throttle)
    return throttle


@pytest.fixture()
def app() -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")
    test_app.state.session_manager = SessionManager()
    return test_app


@pytest.fixture()
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


async def _login(client: httpx.AsyncClient, username: str, password: str, **extra: str) -> Any:
    return await client.post(
        "/api/v1/auth/token", data={"username": username, "password": password, **extra}
    )


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHash:
    def test_roundtrip_accepts_only_the_right_password(self) -> None:
        stored = hash_password(SYNTHETIC_PASSWORD, **FAST_COST)

        assert verify_password(SYNTHETIC_PASSWORD, stored) is True
        assert verify_password(SYNTHETIC_PASSWORD + "x", stored) is False
        assert verify_password("", stored) is False

    def test_format_is_env_and_compose_safe(self) -> None:
        stored = hash_password(SYNTHETIC_PASSWORD, **FAST_COST)

        assert stored.startswith("scrypt:16:8:1:")
        assert "$" not in stored
        assert SYNTHETIC_PASSWORD not in stored
        assert len(stored.split(":")) == 6

    def test_same_password_gets_a_different_salt_each_time(self) -> None:
        assert hash_password("same", **FAST_COST) != hash_password("same", **FAST_COST)

    @pytest.mark.parametrize(
        "stored",
        [
            "",
            "plaintext",
            "scrypt:16:8:1:onlythree",
            "scrypt:x:8:1:c2FsdA:aGFzaA",
            "bcrypt:16:8:1:c2FsdA:aGFzaA",
            "scrypt:3:8:1:c2FsdA:aGFzaA",  # n is not a power of two
            "scrypt:1073741824:8:1:c2FsdA:aGFzaA",  # absurd cost: refuse instead of hanging
            "scrypt:16:8:1::",
        ],
    )
    def test_malformed_stored_hash_never_verifies(self, stored: str) -> None:
        assert verify_password(SYNTHETIC_PASSWORD, stored) is False

    def test_default_cost_is_not_weaker_than_the_documented_minimum(self) -> None:
        stored = hash_password(SYNTHETIC_PASSWORD)

        _, n, r, p, *_ = stored.split(":")
        assert int(n) >= 2**15
        assert int(r) >= 8
        assert int(p) >= 1
        assert verify_password(SYNTHETIC_PASSWORD, stored) is True


class TestSettingsValidation:
    def test_malformed_password_hash_stops_startup(self) -> None:
        with pytest.raises(ValidationError, match="AUTH_PASSWORD_HASH"):
            _settings(auth_password_hash="not-a-hash")

    def test_username_without_hash_is_incomplete(self) -> None:
        with pytest.raises(ValidationError, match="AUTH_USERNAME"):
            _settings(auth_password_hash="")

    def test_hash_without_username_is_incomplete(self) -> None:
        with pytest.raises(ValidationError, match="AUTH_USERNAME"):
            _settings(auth_username="")

    def test_unconfigured_login_is_valid_settings(self) -> None:
        cfg = _settings(auth_username="", auth_password_hash="")

        assert cfg.auth_username == ""

    def test_tier_is_a_closed_set(self) -> None:
        with pytest.raises(ValidationError):
            _settings(auth_tier="root")


# ---------------------------------------------------------------------------
# POST /auth/token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["production", "staging", "development"])
async def test_valid_credentials_issue_a_token_with_server_side_identity(
    environment: str, client: httpx.AsyncClient
) -> None:
    cfg = _settings(environment=environment)

    with _using(cfg):
        response = await _login(client, "ana", SYNTHETIC_PASSWORD)

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 24 * 3600
    claims = jwt.decode(
        body["access_token"],
        SYNTHETIC_SECRET,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
    )
    assert claims["sub"] == "ana"
    assert claims["tenant_id"] == "tenant-ana"
    assert claims["tier"] == "pro"
    assert claims["is_admin"] is False


async def test_issued_token_authenticates_api_calls(client: httpx.AsyncClient) -> None:
    cfg = _settings()
    with _using(cfg):
        token = (await _login(client, "ana", SYNTHETIC_PASSWORD)).json()["access_token"]
        response = await client.get(
            "/api/v1/sessions/00000000-0000-4000-8000-000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 404  # authenticated, session simply does not exist


async def test_request_cannot_choose_tenant_tier_or_admin(client: httpx.AsyncClient) -> None:
    cfg = _settings()

    with _using(cfg):
        response = await _login(
            client,
            "ana",
            SYNTHETIC_PASSWORD,
            tenant_id="victim-tenant",
            tier="enterprise",
            is_admin="true",
        )

    claims = jwt.decode(
        response.json()["access_token"],
        SYNTHETIC_SECRET,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
    )
    assert (claims["tenant_id"], claims["tier"], claims["is_admin"]) == ("tenant-ana", "pro", False)


@pytest.mark.parametrize("environment", ["production", "staging", "development"])
@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("ana", "wrong-password"),
        ("someone-else", SYNTHETIC_PASSWORD),
        ("ANA", SYNTHETIC_PASSWORD),
        ("ana", ""),
        ("", SYNTHETIC_PASSWORD),
        ("", ""),
        ("ana", SYNTHETIC_PASSWORD + " "),
    ],
    ids=[
        "bad_password",
        "unknown_user",
        "case_differs",
        "empty_pw",
        "empty_user",
        "both_empty",
        "extra_space",
    ],
)
async def test_invalid_credentials_get_401_and_no_token(
    environment: str, username: str, password: str, client: httpx.AsyncClient
) -> None:
    cfg = _settings(environment=environment)

    with _using(cfg):
        response = await _login(client, username, password)

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password."}
    assert response.headers["www-authenticate"] == "Bearer"
    assert "access_token" not in response.text


async def test_wrong_user_and_wrong_password_are_indistinguishable(
    client: httpx.AsyncClient,
) -> None:
    cfg = _settings()

    with _using(cfg):
        wrong_pw = await _login(client, "ana", "nope")
        wrong_user = await _login(client, "nobody", SYNTHETIC_PASSWORD)

    assert wrong_pw.status_code == wrong_user.status_code == 401
    assert wrong_pw.json() == wrong_user.json()


async def test_unknown_user_still_pays_for_a_hash_verification(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real = auth_module.verify_password

    def spy(password: str, stored: str) -> bool:
        calls.append(stored)
        return real(password, stored)

    monkeypatch.setattr(auth_module, "verify_password", spy)
    cfg = _settings()

    with _using(cfg):
        await _login(client, "nobody", "whatever")

    assert len(calls) == 1, "unknown users must not be told apart by response time"


@pytest.mark.parametrize("environment", ["production", "staging"])
async def test_unconfigured_login_is_503_not_501_and_issues_nothing(
    environment: str, client: httpx.AsyncClient
) -> None:
    cfg = _settings(environment=environment, auth_username="", auth_password_hash="")

    with _using(cfg):
        response = await _login(client, "ana", SYNTHETIC_PASSWORD)

    assert response.status_code == 503
    assert response.json() == {"detail": "Login is not configured on this server."}


async def test_development_without_config_keeps_the_open_login_for_non_empty_input(
    client: httpx.AsyncClient,
) -> None:
    cfg = _settings(environment="development", auth_username="", auth_password_hash="")

    with _using(cfg):
        ok = await _login(client, "dev", "anything")
        empty_user = await _login(client, "", "anything")
        empty_password = await _login(client, "dev", "")

    assert ok.status_code == 200
    assert empty_user.status_code == 401
    assert empty_password.status_code == 401


async def test_development_with_config_enforces_it(client: httpx.AsyncClient) -> None:
    cfg = _settings(environment="development")

    with _using(cfg):
        response = await _login(client, "dev", "anything")

    assert response.status_code == 401


async def test_oversized_username_is_rejected(client: httpx.AsyncClient) -> None:
    cfg = _settings()

    with _using(cfg):
        response = await _login(client, "u" * 5000, SYNTHETIC_PASSWORD)

    assert response.status_code == 401


async def test_failed_logins_never_log_the_password(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _settings()

    with caplog.at_level(logging.DEBUG), _using(cfg):
        await _login(client, "ana", "attempted-secret-value")

    assert "attempted-secret-value" not in caplog.text


async def test_admin_flag_comes_from_configuration(client: httpx.AsyncClient) -> None:
    cfg = _settings(auth_is_admin=True)

    with _using(cfg):
        token = (await _login(client, "ana", SYNTHETIC_PASSWORD)).json()["access_token"]

    claims = jwt.decode(token, SYNTHETIC_SECRET, algorithms=[JWT_ALGORITHM], audience=JWT_AUDIENCE)
    assert claims["is_admin"] is True


# ---------------------------------------------------------------------------
# Brute-force throttle
# ---------------------------------------------------------------------------


async def test_repeated_failures_lock_the_username_even_for_the_right_password(
    client: httpx.AsyncClient,
) -> None:
    cfg = _settings()

    with _using(cfg):
        statuses = [(await _login(client, "ana", "bad")).status_code for _ in range(5)]
        locked = await _login(client, "ana", SYNTHETIC_PASSWORD)

    assert statuses == [401] * 5
    assert locked.status_code == 429
    assert locked.json() == {"detail": "Too many failed login attempts. Try again later."}
    assert 1 <= int(locked.headers["retry-after"]) <= 60


async def test_lockout_is_per_username(client: httpx.AsyncClient) -> None:
    cfg = _settings()

    with _using(cfg):
        for _ in range(5):
            await _login(client, "mallory", "bad")
        other = await _login(client, "ana", SYNTHETIC_PASSWORD)

    assert other.status_code == 200


async def test_success_clears_the_failure_count(client: httpx.AsyncClient) -> None:
    cfg = _settings()

    with _using(cfg):
        for _ in range(4):
            await _login(client, "ana", "bad")
        assert (await _login(client, "ana", SYNTHETIC_PASSWORD)).status_code == 200
        again = [(await _login(client, "ana", "bad")).status_code for _ in range(4)]

    assert again == [401] * 4


class TestLoginThrottle:
    def test_lock_expires_after_the_window(self) -> None:
        now = [1000.0]
        throttle = LoginThrottle(max_failures=2, lockout_seconds=30, clock=lambda: now[0])
        throttle.record_failure("ana")
        throttle.record_failure("ana")
        assert throttle.retry_after("ana") == 30

        now[0] += 31

        assert throttle.retry_after("ana") is None

    def test_old_failures_do_not_accumulate(self) -> None:
        now = [0.0]
        throttle = LoginThrottle(max_failures=2, lockout_seconds=30, clock=lambda: now[0])
        throttle.record_failure("ana")
        now[0] += 31
        throttle.record_failure("ana")

        assert throttle.retry_after("ana") is None

    def test_tracked_usernames_are_bounded(self) -> None:
        throttle = LoginThrottle(max_failures=3, lockout_seconds=60, max_tracked=100)

        for index in range(1000):
            throttle.record_failure(f"random-{index}")

        assert throttle.tracked_count() <= 100

    def test_username_is_normalised_and_truncated_for_tracking(self) -> None:
        throttle = LoginThrottle(max_failures=1, lockout_seconds=60)
        throttle.record_failure("ANA")

        assert throttle.retry_after("ana") is not None
        throttle.record_failure("x" * 10_000)
        assert throttle.retry_after("x" * 10_000) is not None
