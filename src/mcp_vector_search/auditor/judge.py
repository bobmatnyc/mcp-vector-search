"""LLM-based judge for evaluating privacy claim evidence.

Task 7: Determines PASS/FAIL/INSUFFICIENT_EVIDENCE/MANUAL_REVIEW status for
each claim. Applies deterministic guards first, then calls the configured LLM
backend as judge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .config import AuditorSettings
from .models import Evidence, PolicyClaim, Verdict
from .policy_extractor import _get_model

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_verdict.md"


def _load_judge_prompt() -> str:
    """Load the judge prompt template from disk."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _format_evidence_list(evidence: list[Evidence]) -> str:
    """Format evidence list as readable text for the judge prompt."""
    if not evidence:
        return "No evidence found."

    lines: list[str] = []
    for i, ev in enumerate(evidence, 1):
        lines.append(f"### Evidence {i}: [{ev.tool}]")
        lines.append(f"**File:** {ev.file_path} (lines {ev.start_line}-{ev.end_line})")
        lines.append(f"**Query:** {ev.query}")
        lines.append(f"**Score:** {ev.score:.3f}")
        if ev.kg_path:
            lines.append(f"**KG Path:** {' -> '.join(ev.kg_path)}")
        lines.append(f"**Snippet:**\n```\n{ev.snippet[:500]}\n```")
        lines.append("")

    return "\n".join(lines)


def _build_judge_prompt(claim: PolicyClaim, evidence: list[Evidence]) -> str:
    """Render the judge prompt with claim and evidence injected."""
    template = _load_judge_prompt()
    evidence_text = _format_evidence_list(evidence)

    prompt = (
        template.replace("{{ claim.category }}", claim.category)
        .replace("{{ claim.normalized }}", claim.normalized)
        .replace("{{ claim.text }}", claim.text)
        .replace("{{ evidence_list }}", evidence_text)
    )
    return prompt


def _parse_judge_response(content: list[Any]) -> tuple[str, float, str]:
    """Parse the judge's structured response from Anthropic tool-use content blocks.

    Returns:
        Tuple of (status, confidence, reasoning).
        Defaults to INSUFFICIENT_EVIDENCE on parse failure.
    """
    for block in content:
        block_type = getattr(block, "type", None)

        if (
            block_type == "tool_use"
            and getattr(block, "name", None) == "submit_verdict"
        ):
            inp = block.input
            status = inp.get("status", "INSUFFICIENT_EVIDENCE")
            confidence = float(inp.get("confidence", 0.5))
            reasoning = inp.get("reasoning", "No reasoning provided.")
            return status, confidence, reasoning

        if block_type == "text":
            text = getattr(block, "text", "").strip()
            # Try JSON extraction from text response
            if "{" in text:
                try:
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    obj = json.loads(text[start:end])
                    status = obj.get("status", "INSUFFICIENT_EVIDENCE")
                    confidence = float(obj.get("confidence", 0.5))
                    reasoning = obj.get("reasoning", "No reasoning provided.")
                    return status, confidence, reasoning
                except (ValueError, json.JSONDecodeError):
                    pass

    logger.warning(
        "Could not parse judge response — defaulting to INSUFFICIENT_EVIDENCE"
    )
    return "INSUFFICIENT_EVIDENCE", 0.5, "Failed to parse judge response."


def _parse_judge_json_text(text: str) -> tuple[str, float, str]:
    """Parse a verdict from plain JSON text (OpenRouter path).

    Returns:
        Tuple of (status, confidence, reasoning).
        Defaults to INSUFFICIENT_EVIDENCE on parse failure.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            if part.startswith("json"):
                part = part[4:]
            text = part.strip()
            break

    # Extract first { ... } block
    if "{" in text:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            status = obj.get("status", "INSUFFICIENT_EVIDENCE")
            confidence = float(obj.get("confidence", 0.5))
            reasoning = obj.get("reasoning", "No reasoning provided.")
            return status, confidence, reasoning
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not parse judge JSON response: %s", exc)

    logger.warning(
        "Could not parse judge response text — defaulting to INSUFFICIENT_EVIDENCE"
    )
    return "INSUFFICIENT_EVIDENCE", 0.5, "Failed to parse judge response."


async def _judge_claim_anthropic(
    claim: PolicyClaim,
    evidence: list[Evidence],
    settings: AuditorSettings,
) -> tuple[str, float, str]:
    """Call Anthropic API with tool-use to get a structured verdict.

    Returns:
        Tuple of (status, confidence, reasoning).

    Raises:
        ImportError: If anthropic package is not installed.
        Exception: On API errors (caller handles with fallback verdict).
    """
    import anthropic

    api_key = settings.anthropic_api_key.get_secret_value()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    verdict_tool = {
        "name": "submit_verdict",
        "description": "Submit the compliance verdict for a privacy policy claim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "MANUAL_REVIEW"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["status", "confidence", "reasoning"],
        },
    }

    model = _get_model(settings, "judge")
    prompt = _build_judge_prompt(claim, evidence)

    logger.debug(
        "Judging claim %s (%s) with Anthropic model=%s, evidence_count=%d",
        claim.id,
        claim.category,
        model,
        len(evidence),
    )

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[verdict_tool],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": prompt}],
    )

    return _parse_judge_response(response.content)


async def _judge_claim_openrouter(
    claim: PolicyClaim,
    evidence: list[Evidence],
    settings: AuditorSettings,
) -> tuple[str, float, str]:
    """Call OpenRouter API to get a JSON verdict.

    Returns:
        Tuple of (status, confidence, reasoning).

    Raises:
        ImportError: If openai package is not installed.
        Exception: On API errors (caller handles with fallback verdict).
    """
    from openai import AsyncOpenAI

    api_key = (
        settings.openrouter_api_key.get_secret_value()
        if settings.openrouter_api_key
        else ""
    )
    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=api_key,
    )

    model = _get_model(settings, "judge")
    prompt = _build_judge_prompt(claim, evidence)

    json_instruction = (
        "\n\nRespond with ONLY a JSON object with these fields: "
        "status (one of: PASS, FAIL, INSUFFICIENT_EVIDENCE, MANUAL_REVIEW), "
        "confidence (float 0.0-1.0), "
        "reasoning (string explaining your verdict). "
        "Do not include any explanation or markdown — return raw JSON only."
    )

    logger.debug(
        "Judging claim %s (%s) with OpenRouter model=%s, evidence_count=%d",
        claim.id,
        claim.category,
        model,
        len(evidence),
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt + json_instruction}],
        max_tokens=1024,
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content or ""
    return _parse_judge_json_text(raw_text)


async def judge_claim(
    claim: PolicyClaim,
    evidence: list[Evidence],
    settings: AuditorSettings,
) -> Verdict:
    """Judge a policy claim against collected evidence.

    Applies deterministic guards first, then invokes the LLM judge if needed.

    Deterministic guard order:
    1. Non-testable claims → MANUAL_REVIEW (confidence=1.0)
    2. Insufficient evidence count → INSUFFICIENT_EVIDENCE
    3. KG path required but absent → INSUFFICIENT_EVIDENCE
    4. LLM judge → PASS/FAIL + confidence
    5. Low confidence downgrade → MANUAL_REVIEW

    Args:
        claim: The PolicyClaim being judged.
        evidence: Collected evidence for this claim.
        settings: Auditor settings.

    Returns:
        Verdict object with status, confidence, reasoning, and evidence.
    """
    kg_path_present = any(ev.kg_path for ev in evidence)

    # Guard 1: Non-testable → always MANUAL_REVIEW
    if not claim.testable:
        return Verdict(
            claim_id=claim.id,
            status="MANUAL_REVIEW",
            confidence=1.0,
            reasoning="Claim is marked as non-testable — requires human review.",
            evidence=evidence,
            kg_path_present=kg_path_present,
            evidence_count=len(evidence),
        )

    # Guard 2: Insufficient evidence count
    if len(evidence) < settings.min_evidence_count:
        return Verdict(
            claim_id=claim.id,
            status="INSUFFICIENT_EVIDENCE",
            confidence=1.0,
            reasoning=(
                f"Only {len(evidence)} evidence item(s) found; "
                f"minimum required is {settings.min_evidence_count}."
            ),
            evidence=evidence,
            kg_path_present=kg_path_present,
            evidence_count=len(evidence),
        )

    # Guard 3: KG path required but absent
    if settings.require_kg_path and not kg_path_present:
        return Verdict(
            claim_id=claim.id,
            status="INSUFFICIENT_EVIDENCE",
            confidence=1.0,
            reasoning=(
                "Knowledge graph path is required (require_kg_path=True) "
                "but no KG path was found in the evidence."
            ),
            evidence=evidence,
            kg_path_present=False,
            evidence_count=len(evidence),
        )

    # Guard 4+5: LLM judge — check API key availability
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

    if not has_api_key:
        logger.warning(
            "No API key configured for backend=%s — defaulting to INSUFFICIENT_EVIDENCE for claim %s",
            settings.llm_backend,
            claim.id,
        )
        return Verdict(
            claim_id=claim.id,
            status="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning=f"API key not configured for backend '{settings.llm_backend}' — LLM judge could not be invoked.",
            evidence=evidence,
            kg_path_present=kg_path_present,
            evidence_count=len(evidence),
        )

    try:
        if settings.llm_backend == "openrouter":
            try:
                from openai import AsyncOpenAI  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required for the OpenRouter judge backend. "
                    "Install it with: pip install openai"
                ) from exc
            status, confidence, reasoning = await _judge_claim_openrouter(
                claim, evidence, settings
            )
        else:
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The 'anthropic' package is required for LLM judging. "
                    "Install it with: pip install 'mcp-vector-search[auditor]'"
                ) from exc
            status, confidence, reasoning = await _judge_claim_anthropic(
                claim, evidence, settings
            )

    except Exception as exc:
        logger.error("LLM judge API call failed for claim %s: %s", claim.id, exc)
        return Verdict(
            claim_id=claim.id,
            status="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning=f"LLM judge API call failed: {exc}",
            evidence=evidence,
            kg_path_present=kg_path_present,
            evidence_count=len(evidence),
        )

    # Guard 5: Low confidence → downgrade to MANUAL_REVIEW
    if confidence < settings.confidence_threshold and status in ("PASS", "FAIL"):
        logger.debug(
            "Downgrading claim %s from %s to MANUAL_REVIEW (confidence=%.2f < threshold=%.2f)",
            claim.id,
            status,
            confidence,
            settings.confidence_threshold,
        )
        status = "MANUAL_REVIEW"

    # Validate status is a known value
    valid_statuses = {"PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "MANUAL_REVIEW"}
    if status not in valid_statuses:
        logger.warning(
            "LLM returned unknown status '%s' — using INSUFFICIENT_EVIDENCE", status
        )
        status = "INSUFFICIENT_EVIDENCE"

    return Verdict(
        claim_id=claim.id,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence,
        kg_path_present=kg_path_present,
        evidence_count=len(evidence),
    )
