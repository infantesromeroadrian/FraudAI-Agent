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

from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

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
