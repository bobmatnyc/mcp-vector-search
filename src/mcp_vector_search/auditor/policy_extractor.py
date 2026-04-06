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
# Model selection helper
# ---------------------------------------------------------------------------


def _get_model(settings: AuditorSettings, role: str) -> str:
    """Return the correct model name based on backend and role.

    Args:
        settings: Auditor settings.
        role: Either "extractor" or "judge".

    Returns:
        Model identifier string appropriate for the configured backend.
    """
    if settings.llm_backend == "openrouter":
        return (
            settings.openrouter_extractor_model
            if role == "extractor"
            else settings.openrouter_judge_model
        )
    return settings.extractor_model if role == "extractor" else settings.judge_model


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


def _parse_claims_from_json(text: str, settings: AuditorSettings) -> list[PolicyClaim]:
    """Parse claims from a JSON string returned by the OpenRouter backend.

    OpenRouter returns a plain text/JSON response (no tool-use blocks).
    We expect the model to return a JSON array of claim objects or a JSON
    object with a "claims" key.

    Args:
        text: Raw text response from the OpenRouter model.
        settings: Auditor settings (used for max_claims_per_policy guard).

    Returns:
        List of validated PolicyClaim objects.
    """
    claims: list[PolicyClaim] = []
    text = text.strip()

    # Extract JSON — the model may wrap it in markdown code fences
    if "```" in text:
        # Pull out the content between the first ``` and last ```
        parts = text.split("```")
        for part in parts[1::2]:
            # Strip optional language tag (e.g. "json\n[...")
            if part.startswith("json"):
                part = part[4:]
            text = part.strip()
            break

    # Handle both {"claims": [...]} and [...] formats
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first [ ... ] block
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse JSON from OpenRouter extractor response"
                )
                return claims
        else:
            logger.warning("No JSON array found in OpenRouter extractor response")
            return claims

    if isinstance(parsed, dict):
        raw_list = parsed.get("claims", [])
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        logger.warning("Unexpected JSON structure from OpenRouter extractor")
        return claims

    for raw in raw_list:
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

    return claims


# ---------------------------------------------------------------------------
# LLM extractor (internal) — Anthropic path
# ---------------------------------------------------------------------------


async def _extract_claims_anthropic(
    policy_text: str,
    settings: AuditorSettings,
) -> list[PolicyClaim]:
    """Extract claims via Anthropic's tool-use API.

    Args:
        policy_text: Full text of the privacy policy document.
        settings: Auditor settings.

    Returns:
        List of PolicyClaim objects.
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
    model = _get_model(settings, "extractor")

    logger.info(
        "Extracting claims from policy (%d chars) using model=%s",
        len(policy_text),
        model,
    )

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        tools=[submit_claims_tool],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": prompt}],
    )

    claims = _parse_claims_from_response(response.content, settings)
    logger.info("LLM extracted %d claims from policy", len(claims))
    return claims


# ---------------------------------------------------------------------------
# LLM extractor (internal) — OpenRouter path
# ---------------------------------------------------------------------------


async def _extract_claims_openrouter(
    policy_text: str,
    settings: AuditorSettings,
) -> list[PolicyClaim]:
    """Extract claims via OpenRouter's OpenAI-compatible API.

    Uses a JSON-mode prompt since OpenRouter may not support Anthropic-style
    tool-use blocks.

    Args:
        policy_text: Full text of the privacy policy document.
        settings: Auditor settings.

    Returns:
        List of PolicyClaim objects.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for the OpenRouter backend. "
            "Install it with: pip install openai"
        ) from exc

    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Export OPENROUTER_API_KEY or set MVS_AUDIT_OPENROUTER_API_KEY."
        )

    api_key = settings.openrouter_api_key.get_secret_value()
    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=api_key,
    )

    model = _get_model(settings, "extractor")
    prompt = _build_prompt(policy_text, settings.max_claims_per_policy)

    # Augment prompt to request JSON output since we can't use Anthropic tool-use
    json_instruction = (
        "\n\nRespond with ONLY a JSON array of claim objects. "
        "Each object must have: category (one of: data_sharing, encryption, retention, "
        "user_rights, third_party, logging_pii, consent, security, access_control), "
        "text (verbatim quote), normalized (canonical statement), keywords (array of strings), "
        "policy_section (string or null), testable (boolean), "
        "source_offsets (array of two integers [start, end]). "
        "Do not include any explanation or markdown — return raw JSON only."
    )

    logger.info(
        "Extracting claims from policy (%d chars) using OpenRouter model=%s",
        len(policy_text),
        model,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt + json_instruction,
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content or ""
    claims = _parse_claims_from_json(raw_text, settings)
    logger.info("OpenRouter extracted %d claims from policy", len(claims))
    return claims


# ---------------------------------------------------------------------------
# Public LLM extraction entry point (selects backend)
# ---------------------------------------------------------------------------


async def extract_claims_llm(
    policy_text: str,
    settings: AuditorSettings,
) -> list[PolicyClaim]:
    """Extract claims via the configured LLM backend.

    Dispatches to Anthropic or OpenRouter based on settings.llm_backend.

    Args:
        policy_text: Full text of the privacy policy document.
        settings: Auditor settings containing model name and API key.

    Returns:
        List of PolicyClaim objects (up to settings.max_claims_per_policy).
    """
    if settings.llm_backend == "openrouter":
        return await _extract_claims_openrouter(policy_text, settings)
    return await _extract_claims_anthropic(policy_text, settings)


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

    # 2. Determine whether an API key is available for the configured backend
    has_api_key = False
    if settings.llm_backend == "openrouter":
        has_api_key = bool(
            settings.openrouter_api_key
            and settings.openrouter_api_key.get_secret_value()
        )
    else:
        has_api_key = bool(
            settings.anthropic_api_key and settings.anthropic_api_key.get_secret_value()
        )

    # 3. If API key available AND configured to use LLM, enhance
    if has_api_key and settings.use_llm_extraction:
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
