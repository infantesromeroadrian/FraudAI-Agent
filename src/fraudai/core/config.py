"""Application configuration via environment variables."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Placeholder secrets shipped in the code and in .env.example. A production
# process signed with one of these would accept tokens forged by anyone.
_PLACEHOLDER_JWT_SECRETS = frozenset(
    {"fraudai-dev-secret-change-in-production", "change-this-in-production"}
)
_MIN_JWT_SECRET_LENGTH = 32
MIN_SCRAPE_TOKEN_LENGTH = 32
_PASSWORD_HASH_FIELDS = 6  # scrypt:<n>:<r>:<p>:<salt>:<hash>


def is_volatile_sqlite_path(path: str) -> bool:
    """True for SQLite destinations that live only in memory or a throw-away temp file.

    ``:memory:`` is a private in-memory database; an empty name makes SQLite use a
    temporary on-disk database that is deleted on close. Both lose everything on restart.
    The connection does not interpret ``file:`` URIs, so those are ordinary file names.
    """
    return path.strip() in ("", ":memory:")


class Settings(BaseSettings):
    """FraudAI Agent configuration.

    Reads from environment variables and .env file.
    """

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_keep_alive: str = "5m"

    # LLM Provider. "ollama" reaches the personal daemon at ``ollama_host`` through its
    # OpenAI-compatible API (``/v1``); it needs no API key and has no per-token price.
    llm_provider: Literal["anthropic", "groq", "openai", "ollama"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"  # default Groq model
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""
    # Optional tariff in USD per million tokens. Cost is only emitted when BOTH are set:
    # an unknown price is omitted, never reported as zero.
    llm_input_cost_per_mtok_usd: float | None = Field(default=None, ge=0)
    llm_output_cost_per_mtok_usd: float | None = Field(default=None, ge=0)

    # Sandbox (ADR-007). "docker": run sandbox-backed tools in ephemeral containers
    # through the host's Docker. "disabled": no Docker access at all; those tools
    # return an explicit error (the Compose production mode, which has no docker.sock).
    sandbox_mode: Literal["docker", "disabled"] = "docker"
    sandbox_image: str = "fraudai-sandbox:latest"
    sandbox_timeout_seconds: int = 300  # ceiling: tools may ask for less, never more
    sandbox_mem_limit: str = "2g"
    sandbox_cpu_count: float = 2.0
    sandbox_pids_limit: int = 128
    sandbox_max_concurrent: int = 2

    # Auth / JWT
    jwt_secret: str = "fraudai-dev-secret-change-in-production"

    # Local login (D5): one user configured on the server. ``auth_password_hash`` is an
    # scrypt hash (``python -m fraudai.api.auth``), never the password itself.
    auth_username: str = ""
    auth_password_hash: str = ""
    auth_tenant_id: str = "default"
    auth_tier: Literal["free", "pro", "enterprise"] = "enterprise"
    auth_is_admin: bool = False
    login_max_failures: int = Field(default=5, ge=1)
    login_lockout_seconds: int = Field(default=60, ge=1)

    # Metrics scrape credential: a static bearer value valid only for the metrics route.
    # The file takes precedence over the plain value so both services can mount one secret.
    metrics_scrape_token: str = ""
    metrics_scrape_token_file: str = ""

    # Operations
    # Set by the image build (``BUILD_SHA``); served as-is when it is 7-64 lowercase hex.
    build_sha: str = ""
    database_path: str = "data/fraudai.db"
    readiness_interval_seconds: float = Field(default=15.0, gt=0)
    log_format: Literal["json", "text"] = "json"

    # Per-process limits (no shared store).
    rate_limit_requests_per_minute: int = Field(default=60, ge=1)
    max_sessions_per_tenant: int = Field(default=200, ge=1)
    history_window_messages: int = Field(default=40, ge=1)

    # App
    log_level: str = "INFO"
    # Closed set: a typo such as "prod" or "PRODUCTION" must fail at startup
    # instead of silently skipping the production-only checks.
    environment: Literal["development", "staging", "production"] = "development"
    cors_allowed_origins: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @model_validator(mode="after")
    def _require_real_jwt_secret_outside_development(self) -> Self:
        """Refuse to start in staging or production with a placeholder or short JWT secret.

        Anyone who knows the public default could otherwise sign valid tokens.
        """
        if self.environment != "development" and (
            self.jwt_secret in _PLACEHOLDER_JWT_SECRETS
            or len(self.jwt_secret) < _MIN_JWT_SECRET_LENGTH
        ):
            msg = (
                "JWT_SECRET must be set to a random value of at least "
                f"{_MIN_JWT_SECRET_LENGTH} characters when ENVIRONMENT={self.environment}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_durable_database_outside_development(self) -> Self:
        """Staging and production persist sessions and feedback: the path must be a file."""
        if self.environment != "development" and is_volatile_sqlite_path(self.database_path):
            msg = (
                "DATABASE_PATH must be a file path (not :memory: or empty) "
                f"when ENVIRONMENT={self.environment}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_local_login(self) -> Self:
        """The login user is all-or-nothing and the hash must be well formed."""
        if bool(self.auth_username) != bool(self.auth_password_hash):
            msg = "AUTH_USERNAME and AUTH_PASSWORD_HASH must be set together"
            raise ValueError(msg)
        if self.auth_password_hash and (
            not self.auth_password_hash.startswith("scrypt:")
            or len(self.auth_password_hash.split(":")) != _PASSWORD_HASH_FIELDS
        ):
            msg = "AUTH_PASSWORD_HASH must be an scrypt hash made by `python -m fraudai.api.auth`"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_provider_settings(self) -> Self:
        """Ollama needs a reachable-looking base URL and an explicit model."""
        if self.llm_provider == "ollama":
            if not self.ollama_host.strip().lower().startswith(("http://", "https://")):
                msg = "OLLAMA_HOST must be an http(s) URL when LLM_PROVIDER=ollama"
                raise ValueError(msg)
            if not self.llm_model.strip():
                msg = "LLM_MODEL must name the Ollama model when LLM_PROVIDER=ollama"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_tariff_pair(self) -> Self:
        """Input and output prices go together; one alone cannot price a call."""
        input_price = self.llm_input_cost_per_mtok_usd
        output_price = self.llm_output_cost_per_mtok_usd
        if (input_price is None) != (output_price is None):
            msg = (
                "LLM_INPUT_COST_PER_MTOK_USD and LLM_OUTPUT_COST_PER_MTOK_USD must be set together"
            )
            raise ValueError(msg)
        return self


settings = Settings()
