"""Unit tests for auditor models (Task 2).

Tests:
- Claim ID stability (same input → same ID)
- ID uniqueness (different category or text → different ID)
- Pydantic validation of all models
- Edge cases (empty keywords, None policy_section, etc.)
"""

import pytest
from pydantic import ValidationError

from mcp_vector_search.auditor.models import (
    AuditIgnoreEntry,
    Evidence,
    PolicyClaim,
    Verdict,
)

# ---------------------------------------------------------------------------
# PolicyClaim.compute_id
# ---------------------------------------------------------------------------


class TestPolicyClaimComputeId:
    def test_id_stability(self):
        """Same inputs always produce the same ID."""
        id1 = PolicyClaim.compute_id("encryption", "All API calls use TLS 1.2")
        id2 = PolicyClaim.compute_id("encryption", "All API calls use TLS 1.2")
        assert id1 == id2

    def test_id_length(self):
        """ID is exactly 12 hex characters."""
        claim_id = PolicyClaim.compute_id("data_sharing", "We do not sell user data")
        assert len(claim_id) == 12
        assert all(c in "0123456789abcdef" for c in claim_id)

    def test_id_uniqueness_different_category(self):
        """Different category → different ID."""
        id1 = PolicyClaim.compute_id("encryption", "same text")
        id2 = PolicyClaim.compute_id("data_sharing", "same text")
        assert id1 != id2

    def test_id_uniqueness_different_text(self):
        """Different normalized text → different ID."""
        id1 = PolicyClaim.compute_id("encryption", "text A")
        id2 = PolicyClaim.compute_id("encryption", "text B")
        assert id1 != id2

    def test_id_is_hex(self):
        """Returned ID is a valid hex string."""
        claim_id = PolicyClaim.compute_id("retention", "Data deleted after 90 days")
        int(claim_id, 16)  # raises ValueError if not hex


# ---------------------------------------------------------------------------
# PolicyClaim construction
# ---------------------------------------------------------------------------


class TestPolicyClaimModel:
    def _make_claim(self, **overrides) -> PolicyClaim:
        defaults = {
            "id": "abc123def456",
            "category": "encryption",
            "text": "We encrypt all data in transit using TLS.",
            "normalized": "All network communications use TLS 1.2 or higher.",
            "keywords": ["TLS", "HTTPS", "encrypt"],
            "policy_section": "Security",
            "testable": True,
            "source_offsets": (100, 200),
        }
        defaults.update(overrides)
        return PolicyClaim(**defaults)

    def test_valid_claim_creation(self):
        claim = self._make_claim()
        assert claim.id == "abc123def456"
        assert claim.category == "encryption"
        assert claim.testable is True

    def test_optional_policy_section_none(self):
        claim = self._make_claim(policy_section=None)
        assert claim.policy_section is None

    def test_empty_keywords_allowed(self):
        claim = self._make_claim(keywords=[])
        assert claim.keywords == []

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            self._make_claim(category="invalid_category")

    def test_all_valid_categories(self):
        valid_categories = [
            "data_sharing",
            "encryption",
            "retention",
            "user_rights",
            "third_party",
            "logging_pii",
            "consent",
            "security",
            "access_control",
        ]
        for cat in valid_categories:
            claim = self._make_claim(category=cat)
            assert claim.category == cat

    def test_source_offsets_tuple(self):
        claim = self._make_claim(source_offsets=(0, 500))
        assert claim.source_offsets == (0, 500)


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------


class TestEvidenceModel:
    def _make_evidence(self, **overrides) -> Evidence:
        defaults = {
            "tool": "search_code",
            "query": "TLS encryption",
            "file_path": "src/api/client.py",
            "start_line": 10,
            "end_line": 25,
            "snippet": "session = requests.Session()\nsession.verify = True",
            "score": 0.85,
            "kg_path": None,
        }
        defaults.update(overrides)
        return Evidence(**defaults)

    def test_valid_evidence(self):
        ev = self._make_evidence()
        assert ev.tool == "search_code"
        assert ev.score == 0.85

    def test_kg_path_optional(self):
        ev = self._make_evidence(kg_path=None)
        assert ev.kg_path is None

    def test_kg_path_list(self):
        ev = self._make_evidence(
            kg_path=["main -> send_request", "send_request -> post"]
        )
        assert len(ev.kg_path) == 2

    def test_invalid_tool_raises(self):
        with pytest.raises(ValidationError):
            self._make_evidence(tool="invalid_tool")

    def test_all_valid_tools(self):
        valid_tools = [
            "search_code",
            "search_hybrid",
            "kg_query",
            "kg_callers_at_commit",
            "trace_execution_flow",
            "find_smells",
        ]
        for tool in valid_tools:
            ev = self._make_evidence(tool=tool)
            assert ev.tool == tool


# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------


class TestVerdictModel:
    def _make_verdict(self, **overrides) -> Verdict:
        defaults = {
            "claim_id": "abc123def456",
            "status": "PASS",
            "confidence": 0.9,
            "reasoning": "Strong evidence found.",
            "evidence": [],
            "kg_path_present": True,
            "evidence_count": 3,
        }
        defaults.update(overrides)
        return Verdict(**defaults)

    def test_valid_verdict(self):
        v = self._make_verdict()
        assert v.status == "PASS"
        assert v.ignored is False
        assert v.ignore_justification is None

    def test_ignored_verdict(self):
        v = self._make_verdict(
            ignored=True,
            ignore_justification="This is a valid justification text.",
        )
        assert v.ignored is True
        assert v.ignore_justification is not None

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            self._make_verdict(status="UNKNOWN_STATUS")

    def test_all_valid_statuses(self):
        for status in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "MANUAL_REVIEW"):
            v = self._make_verdict(status=status)
            assert v.status == status


# ---------------------------------------------------------------------------
# AuditIgnoreEntry model
# ---------------------------------------------------------------------------


class TestAuditIgnoreEntry:
    def test_valid_entry(self):
        entry = AuditIgnoreEntry(
            claim_id="abc123def456",
            justification="This claim is excluded because we verified manually.",
            approved_by="alice",
        )
        assert entry.claim_id == "abc123def456"
        assert entry.approved_by == "alice"

    def test_justification_too_short_raises(self):
        with pytest.raises(ValidationError):
            AuditIgnoreEntry(
                claim_id="abc123def456",
                justification="Too short",  # < 20 chars
                approved_by="alice",
            )

    def test_justification_exactly_20_chars(self):
        entry = AuditIgnoreEntry(
            justification="A" * 20,  # exactly 20 chars
            approved_by="alice",
        )
        assert len(entry.justification) == 20

    def test_all_fields_optional_except_justification(self):
        entry = AuditIgnoreEntry(
            justification="This is a long enough justification string.",
            approved_by="bob",
        )
        assert entry.claim_id is None
        assert entry.category is None
        assert entry.pattern is None
        assert entry.expires is None

    def test_category_based_entry(self):
        entry = AuditIgnoreEntry(
            category="logging_pii",
            justification="All logging PII is handled at infrastructure level.",
            approved_by="security-team",
        )
        assert entry.category == "logging_pii"
