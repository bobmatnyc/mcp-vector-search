"""Policy claim extractor — hybrid text-analysis + optional LLM enhancement.

Extracts atomic, testable PolicyClaim objects from policy text.

Strategy:
1. Always run text analysis (no API key needed) via text_extractor.
2. If an API key is available AND use_llm_extraction is True, also run the
   LLM-based extractor and merge the results (dedup by normalized text similarity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .config import AuditorSettings
from .models import PolicyClaim
from .text_extractor import extract_claims_text

_PROMPT_PATH = Path(__file__).parent / "prompts" / "extract_claims.md"


# ---------------------------------------------------------------------------
# Prompt helpers (LLM path)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM extractor (internal)
# ---------------------------------------------------------------------------


async def extract_claims_llm(
    policy_text: str,
    settings: AuditorSettings,
) -> list[PolicyClaim]:
    """Extract claims via Claude (tool-use structured output).

    This is the LLM-based path.  Call extract_claims() instead for the hybrid
    strategy that always falls back to text analysis.

    Args:
        policy_text: Full text of the privacy policy document.
        settings: Auditor settings containing model name and API key.

    Returns:
        List of PolicyClaim objects (up to settings.max_claims_per_policy).

    Raises:
        ImportError: If the anthropic package is not installed.
        ValueError: If no API key is configured.
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
            "This is required for LLM-based policy claim extraction."
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
    logger.info("LLM extracted %d claims from policy", len(claims))
    return claims


# ---------------------------------------------------------------------------
# Merge / dedup helpers
# ---------------------------------------------------------------------------


def _word_overlap_ratio(a: str, b: str) -> float:
    """Compute simple word-overlap Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def merge_claims(
    base: list[PolicyClaim],
    extra: list[PolicyClaim],
    similarity_threshold: float = 0.80,
) -> list[PolicyClaim]:
    """Merge two claim lists, deduplicating by normalized text similarity.

    Claims in *extra* that have >similarity_threshold word-overlap with any
    claim already in *base* are dropped.  Claims that are genuinely new are
    appended.

    Args:
        base: Primary claim list (text-extracted).
        extra: Secondary claim list (LLM-extracted).
        similarity_threshold: Overlap ratio above which two claims are considered
            duplicates (default 0.80).

    Returns:
        Merged, deduplicated list.
    """
    merged = list(base)
    base_normalized = [c.normalized for c in merged]

    for candidate in extra:
        is_dup = any(
            _word_overlap_ratio(candidate.normalized, existing) >= similarity_threshold
            for existing in base_normalized
        )
        if not is_dup:
            merged.append(candidate)
            base_normalized.append(candidate.normalized)

    return merged


# ---------------------------------------------------------------------------
# Public API — hybrid entry point
# ---------------------------------------------------------------------------


async def extract_claims(
    policy_text: str,
    settings: AuditorSettings,
) -> list[PolicyClaim]:
    """Extract claims using hybrid approach: text analysis first, LLM enhancement optional.

    1. Always run text analysis (no API key needed).
    2. If API key available AND settings.use_llm_extraction is True, enhance
       with LLM extraction and merge results.

    Args:
        policy_text: Full text of the privacy policy document.
        settings: Auditor settings controlling models and feature flags.

    Returns:
        List of PolicyClaim objects.
    """
    # 1. Always run text analysis (no API key needed)
    text_claims = await extract_claims_text(policy_text)
    claims = text_claims
    used_llm = False

    # 2. If API key available AND configured to use LLM, enhance
    if settings.anthropic_api_key.get_secret_value() and settings.use_llm_extraction:
        try:
            llm_claims = await extract_claims_llm(policy_text, settings)
            claims = merge_claims(text_claims, llm_claims)
            used_llm = True
        except Exception as e:
            logger.warning("LLM extraction failed, using text analysis only: %s", e)

    if used_llm:
        logger.info(
            "Extracted %d claims (text: %d, LLM: %d, merged: %d)",
            len(claims),
            len(text_claims),
            len(llm_claims),  # type: ignore[possibly-undefined]
            len(claims) - len(text_claims),
        )
    else:
        logger.info(
            "Extracted %d claims via text analysis only (no API key or LLM disabled)",
            len(claims),
        )

    return claims
