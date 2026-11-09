"""Application configuration via environment variables."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings

# Placeholder secrets shipped in the code and in .env.example. A production
# process signed with one of these would accept tokens forged by anyone.
_PLACEHOLDER_JWT_SECRETS = frozenset(
    {"fraudai-dev-secret-change-in-production", "change-this-in-production"}
)
_MIN_JWT_SECRET_LENGTH = 32


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

    # LLM Provider
    llm_provider: str = "groq"  # "anthropic" | "groq" | "openai"
    llm_model: str = "llama-3.3-70b-versatile"  # default Groq model
    anthropic_api_key: str = ""
    groq_api_key: str = ""

    # Sandbox (ADR-007). "docker": run sandbox-backed tools in ephemeral containers
    # through the host's Docker. "disabled": no Docker access at all; those tools
    # return an explicit error (the Compose production mode, which has no docker.sock).
    sandbox_mode: Literal["docker", "disabled"] = "docker"
    sandbox_image: str = "fraudai-sandbox:latest"
    sandbox_timeout_seconds: int = 300
    sandbox_mem_limit: str = "2g"
    sandbox_cpu_count: float = 2.0
    sandbox_pids_limit: int = 128
    sandbox_max_concurrent: int = 2

    # Auth / JWT
    jwt_secret: str = "fraudai-dev-secret-change-in-production"

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


settings = Settings()
