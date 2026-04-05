"""Auditor configuration via pydantic-settings.

Task 3: AuditorSettings loaded from environment variables with MVS_AUDIT_ prefix.
"""

from __future__ import annotations

import os

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuditorSettings(BaseSettings):
    """Settings for the privacy-policy auditor.

    All settings can be overridden via environment variables with the
    MVS_AUDIT_ prefix (e.g., MVS_AUDIT_EXTRACTOR_MODEL=claude-haiku-4-5).

    The anthropic_api_key falls back to the standard ANTHROPIC_API_KEY
    environment variable if MVS_AUDIT_ANTHROPIC_API_KEY is not set.
    """

    model_config = SettingsConfigDict(
        env_prefix="MVS_AUDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: SecretStr = SecretStr("")
    extractor_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-opus-4-6"
    min_evidence_count: int = 2
    require_kg_path: bool = True
    max_claims_per_policy: int = 50
    confidence_threshold: float = 0.7
    use_llm_extraction: bool = True

    @model_validator(mode="before")
    @classmethod
    def resolve_api_key(cls, values: dict) -> dict:
        """Fall back to ANTHROPIC_API_KEY if MVS_AUDIT_ANTHROPIC_API_KEY not set."""
        if not values.get("anthropic_api_key"):
            env_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if env_key:
                values["anthropic_api_key"] = env_key
        return values
