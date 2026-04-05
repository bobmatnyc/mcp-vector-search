"""Auditor configuration via pydantic-settings.

Task 3: AuditorSettings loaded from environment variables with MVS_AUDIT_ prefix.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuditorSettings(BaseSettings):
    """Settings for the privacy-policy auditor.

    All settings can be overridden via environment variables with the
    MVS_AUDIT_ prefix (e.g., MVS_AUDIT_EXTRACTOR_MODEL=claude-haiku-4-5).

    The anthropic_api_key falls back to the standard ANTHROPIC_API_KEY
    environment variable if MVS_AUDIT_ANTHROPIC_API_KEY is not set.

    OpenRouter support:
    - Set MVS_AUDIT_LLM_BACKEND=openrouter to use OpenRouter
    - Set OPENROUTER_API_KEY (or MVS_AUDIT_OPENROUTER_API_KEY) for the key
    - If OPENROUTER_API_KEY is set but no ANTHROPIC_API_KEY, backend
      automatically switches to "openrouter"
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

    # OpenRouter settings
    openrouter_api_key: SecretStr | None = None
    llm_backend: Literal["anthropic", "openrouter"] = "anthropic"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_extractor_model: str = "anthropic/claude-haiku-4-5"
    openrouter_judge_model: str = "anthropic/claude-sonnet-4-5"

    @model_validator(mode="before")
    @classmethod
    def resolve_api_key(cls, values: dict) -> dict:
        """Fall back to ANTHROPIC_API_KEY if MVS_AUDIT_ANTHROPIC_API_KEY not set.

        Also checks OPENROUTER_API_KEY and auto-selects backend when only
        OpenRouter key is available.
        """
        # Resolve Anthropic key
        if not values.get("anthropic_api_key"):
            env_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if env_key:
                values["anthropic_api_key"] = env_key

        # Resolve OpenRouter key (MVS_AUDIT_ prefix takes priority, then bare env var)
        if not values.get("openrouter_api_key"):
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
            if or_key:
                values["openrouter_api_key"] = or_key

        # Auto-select backend: if only OpenRouter key is available, switch automatically
        if not values.get("llm_backend") or values.get("llm_backend") == "anthropic":
            has_anthropic = bool(values.get("anthropic_api_key"))
            has_openrouter = bool(values.get("openrouter_api_key"))
            if has_openrouter and not has_anthropic:
                values["llm_backend"] = "openrouter"

        return values
