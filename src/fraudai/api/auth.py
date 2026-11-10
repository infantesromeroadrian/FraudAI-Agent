"""Authentication and authorization for FraudAI Agent API.

Implements OAuth2 + JWT with RBAC by tier (Free/Pro/Enterprise) as
required by SR-010. Uses PyJWT with HS256 for the MVP; RS256 planned
for production.

Security requirements addressed:
- SR-010: OAuth2/JWT authentication, RBAC per tier, rate limiting
- SR-009: Tenant isolation via user.tenant_id
- SEC-004: Audit trail (user identity propagated to all handlers)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import math
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fraudai.core.config import MIN_SCRAPE_TOKEN_LENGTH

if TYPE_CHECKING:
    from collections.abc import Callable

    from fraudai.core.config import Settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
# Same scheme, but a missing header is left to the caller (the metrics route has two
# kinds of credential and answers 401 itself).
_optional_bearer = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# JWT configuration — algorithm and expiration.
# Secret is read from settings at call time to respect env overrides.
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
JWT_ISSUER = "fraudai-agent"
JWT_AUDIENCE = "fraudai-api"
# Claims that must be present; PyJWT rejects the token otherwise.
_REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub", "tenant_id", "tier"]


class User(BaseModel):
    """Authenticated user context injected into route handlers."""

    user_id: str
    tenant_id: str
    email: str
    tier: Literal["free", "pro", "enterprise"]
    is_admin: bool = False


class _TokenClaims(BaseModel):
    """Strict shape of the claims we trust after the signature check.

    ``strict`` blocks lax coercion (e.g. ``"yes"`` becoming ``is_admin=True``).
    """

    model_config = ConfigDict(strict=True, extra="ignore")

    sub: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    tier: Literal["free", "pro", "enterprise"]
    email: str = ""
    is_admin: bool = False


_DEV_USER = User(
    user_id="dev-user",
    tenant_id="dev-tenant",
    email="dev@fraudai.local",
    tier="enterprise",
    is_admin=True,
)


def create_token(
    user_id: str,
    tenant_id: str,
    tier: str,
    *,
    email: str = "",
    is_admin: bool = False,
) -> str:
    """Create a signed JWT token.

    Args:
        user_id: Subject claim (``sub``).
        tenant_id: Tenant identifier for isolation (SR-009).
        tier: User subscription tier (``free`` / ``pro`` / ``enterprise``).
        email: Optional email claim.
        is_admin: Whether the user has admin privileges.

    Returns:
        Encoded JWT string.
    """
    from fraudai.core.config import settings

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "tier": tier,
        "email": email,
        "is_admin": is_admin,
        "exp": datetime.now(UTC) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(UTC),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> User:
    """Validate JWT and return the authenticated user.

    Only in the ``development`` environment the literal ``dev-token`` value
    bypasses JWT validation and returns a dev user with enterprise tier.
    Staging and production always validate a real JWT. Any other token is
    validated as a JWT in every environment.

    JWT validation:
    1. Verifies the HS256 signature, ``exp``, ``iss`` and ``aud``.
    2. Requires ``sub``, ``tenant_id`` and ``tier`` with the right types.
    3. Builds the User model from the claims.

    Raises:
        HTTPException 401: Missing, malformed, invalid or expired token.
    """
    from fraudai.core.config import settings

    if settings.environment == "development" and token == "dev-token":
        return _DEV_USER

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": _REQUIRED_CLAIMS},
        )
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("Invalid token") from exc

    try:
        claims = _TokenClaims.model_validate(payload)
    except ValidationError as exc:
        raise _unauthorized("Invalid token") from exc

    return User(
        user_id=claims.sub,
        tenant_id=claims.tenant_id,
        email=claims.email,
        tier=claims.tier,
        is_admin=claims.is_admin,
    )


# ---------------------------------------------------------------------------
# Local login (D5): scrypt password hash, one server-configured user
# ---------------------------------------------------------------------------

# scrypt parameters for new hashes: N=2^15, r=8, p=1 (about 32 MiB, ~100 ms). The cost is
# stored inside the hash, so it can be raised later without invalidating old ones.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_KEY_LENGTH = 32
_SCRYPT_SALT_BYTES = 16
_SCRYPT_MAXMEM = 256 * 1024 * 1024
# Ceilings applied when VERIFYING: a corrupt or hostile stored hash must not be able to
# make the server allocate gigabytes or spin for minutes.
_SCRYPT_MAX_N = 2**20
_SCRYPT_MAX_R = 16
_SCRYPT_MAX_P = 4
MAX_USERNAME_LENGTH = 128


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(
    password: str, *, n: int = _SCRYPT_N, r: int = _SCRYPT_R, p: int = _SCRYPT_P
) -> str:
    """Hash *password* as ``scrypt:<n>:<r>:<p>:<salt>:<hash>`` (no ``$``: safe in .env/Compose)."""
    salt = secrets.token_bytes(_SCRYPT_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_SCRYPT_KEY_LENGTH,
    )
    return f"scrypt:{n}:{r}:{p}:{_b64(salt)}:{_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of *password* against a stored hash; malformed hashes never match."""
    try:
        scheme, n_text, r_text, p_text, salt_text, digest_text = stored.split(":")
        n, r, p = int(n_text), int(r_text), int(p_text)
        salt, expected = _unb64(salt_text), _unb64(digest_text)
    except (ValueError, binascii.Error):
        return False
    if (
        scheme != "scrypt"
        or not salt
        or not expected
        or n < 2
        or n & (n - 1)
        or n > _SCRYPT_MAX_N
        or not 1 <= r <= _SCRYPT_MAX_R
        or not 1 <= p <= _SCRYPT_MAX_P
    ):
        return False
    try:
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            maxmem=_SCRYPT_MAXMEM,
            dklen=len(expected),
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


@dataclass(frozen=True, slots=True)
class LocalCredentials:
    """The single login user configured on the server; identity never comes from the request."""

    username: str
    password_hash: str
    tenant_id: str
    tier: Literal["free", "pro", "enterprise"]
    is_admin: bool


def credentials_from_settings(cfg: Settings) -> LocalCredentials | None:
    """The configured login user, or ``None`` when login is not configured."""
    if not cfg.auth_username or not cfg.auth_password_hash:
        return None
    return LocalCredentials(
        username=cfg.auth_username,
        password_hash=cfg.auth_password_hash,
        tenant_id=cfg.auth_tenant_id,
        tier=cfg.auth_tier,
        is_admin=cfg.auth_is_admin,
    )


# Verified against when the username is unknown, so a wrong username costs the same time
# as a wrong password. Built from a throw-away random password, at default cost.
_DECOY_HASH = hash_password(secrets.token_urlsafe(16))


def check_credentials(creds: LocalCredentials, username: str, password: str) -> bool:
    """Compare a login attempt with the configured user (blocking: run it in a thread)."""
    username_ok = hmac.compare_digest(username.encode("utf-8"), creds.username.encode("utf-8"))
    target = creds.password_hash if username_ok else _DECOY_HASH
    password_ok = verify_password(password, target)
    return username_ok and password_ok


class LoginThrottle:
    """Per-username lockout after repeated failures (in-process, bounded memory).

    ``max_failures`` failures inside ``lockout_seconds`` lock that username for the rest of
    the window, even for the right password. Anyone can therefore lock the owner out for a
    minute by guessing; that is the price of not depending on a client address behind a proxy.
    """

    def __init__(
        self,
        max_failures: int = 5,
        lockout_seconds: int = 60,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_tracked: int = 1024,
    ) -> None:
        self._max_failures = max_failures
        self._lockout = lockout_seconds
        self._clock = clock
        self._max_tracked = max_tracked
        self._failures: OrderedDict[str, list[float]] = OrderedDict()

    @staticmethod
    def _key(username: str) -> str:
        return username.strip().lower()[:MAX_USERNAME_LENGTH]

    def _recent(self, key: str) -> list[float]:
        cutoff = self._clock() - self._lockout
        recent = [t for t in self._failures.get(key, []) if t > cutoff]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def retry_after(self, username: str) -> int | None:
        """Seconds until the username may try again, or ``None`` if it is not locked."""
        recent = self._recent(self._key(username))
        if len(recent) < self._max_failures:
            return None
        return max(1, math.ceil(recent[0] + self._lockout - self._clock()))

    def record_failure(self, username: str) -> None:
        key = self._key(username)
        recent = self._recent(key)
        recent.append(self._clock())
        self._failures[key] = recent
        self._failures.move_to_end(key)
        while len(self._failures) > self._max_tracked:
            self._failures.popitem(last=False)

    def reset(self, username: str) -> None:
        self._failures.pop(self._key(username), None)

    def tracked_count(self) -> int:
        return len(self._failures)


# Defaults match the settings defaults; ``configure_login_throttle`` rebuilds it from the
# real settings when the application starts. Routes read it through this module so tests
# (and that rebuild) can replace it.
login_throttle = LoginThrottle()


def configure_login_throttle(cfg: Settings) -> LoginThrottle:
    """Rebuild the module throttle from settings (called once at application start)."""
    global login_throttle  # noqa: PLW0603
    login_throttle = LoginThrottle(cfg.login_max_failures, cfg.login_lockout_seconds)
    return login_throttle


# ---------------------------------------------------------------------------
# Metrics scrape credential (valid for the metrics route only)
# ---------------------------------------------------------------------------


def load_scrape_token(cfg: Settings) -> str | None:
    """The configured scrape credential, or ``None`` if absent, unreadable or too short."""
    value = cfg.metrics_scrape_token
    if cfg.metrics_scrape_token_file:
        try:
            value = Path(cfg.metrics_scrape_token_file).read_text(encoding="utf-8")
        except OSError:
            logger.error("METRICS_SCRAPE_TOKEN_FILE is not readable; scrape credential disabled")
            return None
    value = value.strip()
    if len(value) < MIN_SCRAPE_TOKEN_LENGTH:
        if value:
            logger.error(
                "Metrics scrape credential shorter than %d characters ignored",
                MIN_SCRAPE_TOKEN_LENGTH,
            )
        return None
    return value


async def require_metrics_access(
    token: str | None = Depends(_optional_bearer),
) -> None:
    """Allow the dedicated scrape credential or an admin JWT; nothing else.

    The scrape credential is not a JWT, so every other route rejects it, and it carries no
    identity: it can read the metrics and do nothing else.

    Raises:
        HTTPException 401: no/invalid credential. 403: valid JWT that is not admin.
    """
    from fraudai.core.config import settings

    if not token:
        raise _unauthorized("Not authenticated")
    scrape = load_scrape_token(settings)
    if scrape is not None and hmac.compare_digest(token.encode("utf-8"), scrape.encode("utf-8")):
        return
    user = await get_current_user(token)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )


async def get_admin_user(
    user: User = Depends(get_current_user),
) -> User:
    """Require admin privileges for sensitive endpoints (e.g. /admin/metrics).

    Raises:
        HTTPException 403: User is not an admin.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return user


def _main() -> None:  # pragma: no cover - interactive helper
    """Print an ``AUTH_PASSWORD_HASH=...`` line for a password typed without echo."""
    import getpass
    import sys

    first = getpass.getpass("Password: ")
    if not first or getpass.getpass("Repeat password: ") != first:
        sys.exit("Passwords are empty or do not match; nothing printed.")
    sys.stdout.write(f"AUTH_PASSWORD_HASH={hash_password(first)}\n")


if __name__ == "__main__":  # pragma: no cover
    _main()
