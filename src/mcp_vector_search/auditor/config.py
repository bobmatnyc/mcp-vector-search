"""Auditor configuration via pydantic-settings.

Task 3: AuditorSettings loaded from environment variables with MVS_AUDIT_ prefix.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuditorSettings(BaseSettings):
    """Settings for the privacy-policy auditor.

    All settings can be overridden via environment variables with the
    MVS_AUDIT_ prefix (e.g., MVS_AUDIT_EXTRACTOR_MODEL=claude-haiku-4-5).
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
