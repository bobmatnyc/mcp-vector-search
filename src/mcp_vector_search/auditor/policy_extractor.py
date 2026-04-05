"""Policy claim extractor using Anthropic Claude (haiku model).

Task 4: Extracts atomic, testable PolicyClaim objects from policy text
using structured output (tool-use) via the Anthropic SDK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .config import AuditorSettings
from .models import PolicyClaim

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract_claims.md"


def _load_prompt_template() -> str:
    """Load the extraction prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_prompt(policy_text: str, max_claims: int) -> str:
    """Render the extraction prompt with the policy text injected."""
    template = _load_prompt_template()
    return template.replace("{{ policy_text }}", policy_text).replace(
        "{{ max_claims }}", str(max_claims)
    )


def _parse_claims_from_response(
    content: list[Any], settings: AuditorSettings
) -> list[PolicyClaim]:
    """Parse claim objects from Anthropic tool-use response content blocks.

    Args:
        content: List of content blocks from the Anthropic API response.
        settings: Auditor settings (used for max_claims_per_policy guard).

    Returns:
        List of validated PolicyClaim objects.
    """
    claims: list[PolicyClaim] = []

    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use" and getattr(block, "name", None) == "submit_claims":
            raw_claims = block.input.get("claims", [])
            for raw in raw_claims:
                if len(claims) >= settings.max_claims_per_policy:
                    logger.warning(
                        "Reached max_claims_per_policy=%d — truncating",
                        settings.max_claims_per_policy,
                    )
                    break
                try:
                    claim_id = PolicyClaim.compute_id(
                        raw.get("category", ""),
                        raw.get("normalized", ""),
                    )
                    offsets = raw.get("source_offsets", [0, 0])
                    claim = PolicyClaim(
                        id=claim_id,
                        category=raw["category"],
                        text=raw.get("text", ""),
                        normalized=raw.get("normalized", ""),
                        keywords=raw.get("keywords", []),
                        policy_section=raw.get("policy_section"),
                        testable=raw.get("testable", True),
                        source_offsets=(offsets[0], offsets[1]),
                    )
                    claims.append(claim)
                except (KeyError, ValueError) as exc:
                    logger.warning("Skipping malformed claim: %s — %s", raw, exc)
        elif block_type == "text":
            # Try parsing JSON fallback from text blocks
            text = getattr(block, "text", "")
            if text.strip().startswith("["):
                try:
                    raw_list = json.loads(text)
                    for raw in raw_list:
                        if len(claims) >= settings.max_claims_per_policy:
                            break
                        claim_id = PolicyClaim.compute_id(
                            raw.get("category", ""),
                            raw.get("normalized", ""),
                        )
                        offsets = raw.get("source_offsets", [0, 0])
                        claim = PolicyClaim(
                            id=claim_id,
                            category=raw["category"],
                            text=raw.get("text", ""),
                            normalized=raw.get("normalized", ""),
                            keywords=raw.get("keywords", []),
                            policy_section=raw.get("policy_section"),
                            testable=raw.get("testable", True),
                            source_offsets=(offsets[0], offsets[1]),
                        )
                        claims.append(claim)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    return claims


async def extract_claims(
    policy_text: str,
    settings: AuditorSettings,
) -> list[PolicyClaim]:
    """Extract atomic, testable claims from a privacy policy.

    Uses Claude Haiku with tool-use (structured output) to parse the policy
    and return a list of PolicyClaim objects.

    Args:
        policy_text: Full text of the privacy policy document.
        settings: Auditor settings containing model name and API key.

    Returns:
        List of PolicyClaim objects (up to settings.max_claims_per_policy).

    Raises:
        ImportError: If the anthropic package is not installed.
        Exception: On Anthropic API errors.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for claim extraction. "
            "Install it with: pip install 'mcp-vector-search[auditor]'"
        ) from exc

    api_key = settings.anthropic_api_key.get_secret_value()
    if not api_key:
        raise ValueError(
            "MVS_AUDIT_ANTHROPIC_API_KEY environment variable is not set. "
            "This is required for policy claim extraction."
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Define the tool for structured output
    submit_claims_tool = {
        "name": "submit_claims",
        "description": "Submit the extracted atomic privacy claims as structured data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [
                                    "data_sharing",
                                    "encryption",
                                    "retention",
                                    "user_rights",
                                    "third_party",
                                    "logging_pii",
                                    "consent",
                                    "security",
                                    "access_control",
                                ],
                            },
                            "text": {"type": "string"},
                            "normalized": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "policy_section": {"type": "string"},
                            "testable": {"type": "boolean"},
                            "source_offsets": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "required": [
                            "category",
                            "text",
                            "normalized",
                            "keywords",
                            "testable",
                            "source_offsets",
                        ],
                    },
                }
            },
            "required": ["claims"],
        },
    }

    prompt = _build_prompt(policy_text, settings.max_claims_per_policy)

    logger.info(
        "Extracting claims from policy (%d chars) using model=%s",
        len(policy_text),
        settings.extractor_model,
    )

    response = await client.messages.create(
        model=settings.extractor_model,
        max_tokens=4096,
        tools=[submit_claims_tool],
        tool_choice={"type": "auto"},
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    claims = _parse_claims_from_response(response.content, settings)
    logger.info("Extracted %d claims from policy", len(claims))
    return claims
