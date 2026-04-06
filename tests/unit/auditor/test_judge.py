"""Unit tests for the claim judge (Task 7).

Tests:
- Deterministic guard: non-testable → MANUAL_REVIEW
- Deterministic guard: insufficient evidence count → INSUFFICIENT_EVIDENCE
- Deterministic guard: KG path required but absent → INSUFFICIENT_EVIDENCE
- LLM confidence downgrade: low confidence → MANUAL_REVIEW
- Mocked LLM response parsing
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vector_search.auditor.config import AuditorSettings
from mcp_vector_search.auditor.judge import judge_claim
from mcp_vector_search.auditor.models import Evidence, PolicyClaim


def _make_settings(**overrides) -> AuditorSettings:
    defaults = {
        "anthropic_api_key": "test-key-not-real-x" * 2,
        "extractor_model": "claude-haiku-4-5",
        "judge_model": "claude-opus-4-6",
        "min_evidence_count": 2,
        "require_kg_path": True,
        "confidence_threshold": 0.7,
    }
    defaults.update(overrides)
    # Bypass env loading for tests
    with patch.dict(
        "os.environ",
        {
            "MVS_AUDIT_ANTHROPIC_API_KEY": defaults["anthropic_api_key"],
            "MVS_AUDIT_MIN_EVIDENCE_COUNT": str(defaults["min_evidence_count"]),
            "MVS_AUDIT_REQUIRE_KG_PATH": str(defaults["require_kg_path"]).lower(),
            "MVS_AUDIT_CONFIDENCE_THRESHOLD": str(defaults["confidence_threshold"]),
        },
    ):
        return AuditorSettings()


def _make_claim(testable: bool = True, category: str = "encryption") -> PolicyClaim:
    return PolicyClaim(
        id="abc123def456",
        category=category,
        text="We encrypt data in transit using TLS.",
        normalized="All network communications use TLS 1.2 or higher.",
        keywords=["TLS", "HTTPS"],
        policy_section="Security",
        testable=testable,
        source_offsets=(0, 100),
    )


def _make_evidence(
    count: int = 2,
    with_kg_path: bool = True,
) -> list[Evidence]:
    evidence = []
    for i in range(count):
        evidence.append(
            Evidence(
                tool="search_code",
                query="TLS encryption",
                file_path=f"src/api/client_{i}.py",
                start_line=10 * i,
                end_line=10 * i + 20,
                snippet="requests.get(url, verify=True)",
                score=0.85,
                kg_path=["main -> send_request"] if (with_kg_path and i == 0) else None,
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# Deterministic guard tests (no LLM needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_testable_claim_returns_manual_review():
    """Non-testable claims always get MANUAL_REVIEW without calling LLM."""
    claim = _make_claim(testable=False)
    evidence = _make_evidence(count=5, with_kg_path=True)
    settings = _make_settings()

    verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "MANUAL_REVIEW"
    assert verdict.confidence == 1.0
    assert "non-testable" in verdict.reasoning.lower()


@pytest.mark.asyncio
async def test_insufficient_evidence_count():
    """Too few evidence items → INSUFFICIENT_EVIDENCE without LLM."""
    claim = _make_claim(testable=True)
    evidence = _make_evidence(count=1, with_kg_path=True)  # only 1, min is 2
    settings = _make_settings(min_evidence_count=2)

    verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "INSUFFICIENT_EVIDENCE"
    assert verdict.evidence_count == 1


@pytest.mark.asyncio
async def test_zero_evidence_is_insufficient():
    """Zero evidence → INSUFFICIENT_EVIDENCE."""
    claim = _make_claim(testable=True)
    evidence = []
    settings = _make_settings(min_evidence_count=2)

    verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_kg_path_required_but_absent():
    """KG path required but no evidence has kg_path → INSUFFICIENT_EVIDENCE."""
    claim = _make_claim(testable=True)
    evidence = _make_evidence(count=3, with_kg_path=False)
    settings = _make_settings(require_kg_path=True)

    verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "INSUFFICIENT_EVIDENCE"
    assert verdict.kg_path_present is False


@pytest.mark.asyncio
async def test_kg_path_not_required_proceeds_to_llm():
    """When require_kg_path=False, missing KG path should not block LLM call."""
    claim = _make_claim(testable=True)
    evidence = _make_evidence(count=3, with_kg_path=False)
    settings = _make_settings(require_kg_path=False)

    # Mock the LLM call
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "submit_verdict"
    mock_block.input = {
        "status": "PASS",
        "confidence": 0.9,
        "reasoning": "Evidence is strong.",
    }
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "PASS"


# ---------------------------------------------------------------------------
# LLM response parsing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_pass_verdict():
    """LLM returns PASS with high confidence → PASS verdict."""
    claim = _make_claim()
    evidence = _make_evidence(count=3, with_kg_path=True)
    settings = _make_settings(require_kg_path=True, confidence_threshold=0.7)

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "submit_verdict"
    mock_block.input = {
        "status": "PASS",
        "confidence": 0.9,
        "reasoning": "TLS is correctly configured in all network calls.",
    }
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "PASS"
    assert verdict.confidence == 0.9


@pytest.mark.asyncio
async def test_llm_fail_verdict():
    """LLM returns FAIL with high confidence → FAIL verdict."""
    claim = _make_claim()
    evidence = _make_evidence(count=3, with_kg_path=True)
    settings = _make_settings(require_kg_path=True, confidence_threshold=0.7)

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "submit_verdict"
    mock_block.input = {
        "status": "FAIL",
        "confidence": 0.88,
        "reasoning": "Found plaintext HTTP calls without TLS.",
    }
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "FAIL"


@pytest.mark.asyncio
async def test_low_confidence_downgrades_to_manual_review():
    """LLM returns PASS with low confidence → downgraded to MANUAL_REVIEW."""
    claim = _make_claim()
    evidence = _make_evidence(count=3, with_kg_path=True)
    settings = _make_settings(require_kg_path=True, confidence_threshold=0.7)

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "submit_verdict"
    mock_block.input = {
        "status": "PASS",
        "confidence": 0.5,  # below threshold of 0.7
        "reasoning": "Some evidence but not conclusive.",
    }
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "MANUAL_REVIEW"
    assert verdict.confidence == 0.5


@pytest.mark.asyncio
async def test_no_api_key_returns_insufficient():
    """Missing API key returns INSUFFICIENT_EVIDENCE without calling LLM."""
    claim = _make_claim()
    evidence = _make_evidence(count=3, with_kg_path=True)

    with patch.dict(
        "os.environ",
        {
            "MVS_AUDIT_ANTHROPIC_API_KEY": "",  # empty
            "MVS_AUDIT_REQUIRE_KG_PATH": "true",
        },
    ):
        settings = AuditorSettings()

    mock_anthropic = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        verdict = await judge_claim(claim, evidence, settings)

    assert verdict.status == "INSUFFICIENT_EVIDENCE"
    assert verdict.confidence == 0.0


@pytest.mark.asyncio
async def test_verdict_includes_evidence_list():
    """Returned verdict should carry the full evidence list."""
    claim = _make_claim()
    evidence = _make_evidence(count=3, with_kg_path=True)
    settings = _make_settings()

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "submit_verdict"
    mock_block.input = {"status": "PASS", "confidence": 0.9, "reasoning": "Strong."}
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic = MagicMock()
    mock_anthropic.AsyncAnthropic = MagicMock(return_value=mock_client)

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        verdict = await judge_claim(claim, evidence, settings)

    assert len(verdict.evidence) == 3
    assert verdict.evidence_count == 3
    assert verdict.kg_path_present is True
