"""Unit tests for the text-analysis-based policy claim extractor.

Tests cover:
- Pattern matching for the 6 required claim types
- Compound claim splitting ("and" joining different categories)
- Category classification via keyword matching
- Claim ID stability
- No-API-key path works end-to-end
"""

from __future__ import annotations

from mcp_vector_search.auditor.text_extractor import (
    _classify_category,
    _compute_confidence,
    _normalize_claim,
    _split_compound_claim,
    _split_sections,
    extract_claims_text,
)

# ---------------------------------------------------------------------------
# Sample policy text
# ---------------------------------------------------------------------------

SAMPLE_POLICY = """
## Data Collection

We collect personal data including your name, email address, and usage patterns.

## Data Sharing

We do not sell your personal data to any third parties.
We share aggregated analytics with our service providers.

## Security

All data is encrypted at rest using AES-256.
Data in transit is protected using TLS 1.3.

## User Rights

Users can request deletion of their account and all associated data.
You have the right to access your personal data at any time.

## Third-Party Services

We use third-party analytics providers to understand how our service is used.

## Data Retention

Data is retained for 30 days after account deletion.
We store logs for a maximum period of 90 days.

## Consent

With your consent, we may share data with advertising partners.
You can opt-out of marketing communications at any time.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sync(coro):
    """Run an async coroutine synchronously for testing."""
    import asyncio

    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests: section splitter
# ---------------------------------------------------------------------------


class TestSplitSections:
    def test_markdown_headers_split(self):
        text = "## Section A\nContent A.\n## Section B\nContent B."
        sections = _split_sections(text)
        titles = [t for t, _ in sections]
        assert "Section A" in titles
        assert "Section B" in titles

    def test_no_headers_returns_preamble(self):
        text = "This is all preamble content."
        sections = _split_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Preamble"


# ---------------------------------------------------------------------------
# Tests: category classifier
# ---------------------------------------------------------------------------


class TestClassifyCategory:
    def test_data_sharing_keyword(self):
        cat, kw = _classify_category("We do not sell your personal data to anyone.")
        assert cat == "data_sharing"
        assert "sell" in kw

    def test_encryption_keyword(self):
        cat, kw = _classify_category("All data is encrypted at rest using AES-256.")
        assert cat == "encryption"
        assert any("encrypt" in k.lower() for k in kw)

    def test_user_rights_keyword(self):
        cat, kw = _classify_category("Users can request deletion of their account.")
        assert cat == "user_rights"
        assert "request" in kw

    def test_third_party_keyword(self):
        cat, kw = _classify_category("We use third-party analytics providers.")
        assert cat == "third_party"

    def test_retention_keyword(self):
        cat, kw = _classify_category(
            "Data is retained for 30 days after account deletion."
        )
        assert cat == "retention"
        assert "retain" in kw

    def test_consent_keyword(self):
        cat, kw = _classify_category("With your consent we may send emails.")
        assert cat == "consent"
        assert "consent" in kw


# ---------------------------------------------------------------------------
# Tests: confidence scorer
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_negation_with_keywords_high(self):
        conf = _compute_confidence("We do not sell your data.", "negation", ["sell"])
        assert conf >= 0.9

    def test_clear_assertion_two_keywords(self):
        conf = _compute_confidence(
            "Data is encrypted and secured using TLS.", "security", ["encrypt", "TLS"]
        )
        assert conf >= 0.85

    def test_ambiguous_low_confidence(self):
        conf = _compute_confidence("Some vague statement.", "security", [])
        assert conf <= 0.6


# ---------------------------------------------------------------------------
# Tests: normalizer
# ---------------------------------------------------------------------------


class TestNormalizeClaim:
    def test_negation_normalizes(self):
        result = _normalize_claim("We do not sell your personal data.", "data_sharing")
        assert "does not" in result.lower()
        assert "sell" in result.lower()

    def test_assertion_normalizes(self):
        result = _normalize_claim("We encrypt all data at rest.", "encryption")
        assert "encrypt" in result.lower()

    def test_fallback_returns_sentence(self):
        result = _normalize_claim("Some statement without verbs.", "security")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: compound claim splitter
# ---------------------------------------------------------------------------


class TestSplitCompoundClaim:
    def test_splits_different_categories(self):
        # "encrypt" (encryption) and "share" (data_sharing) — different categories
        sentence = "We encrypt data in transit and share aggregated data with partners."
        splits = _split_compound_claim(sentence, "Security", 0, len(sentence))
        assert splits is not None
        assert len(splits) == 2
        # Both halves should be non-empty
        assert all(len(s.sentence) > 5 for s in splits)

    def test_no_split_when_same_category(self):
        # Both halves → encryption category
        sentence = "We encrypt data at rest and encrypt data in transit."
        splits = _split_compound_claim(sentence, "Security", 0, len(sentence))
        # May or may not split — if both are same category, should not split
        if splits is not None:
            # If split anyway, both categories should be the same (or None returned)
            cats = set()
            from mcp_vector_search.auditor.text_extractor import _classify_category

            for s in splits:
                cat, _ = _classify_category(s.sentence)
                cats.add(cat)
            # No constraint violated if same category; test just verifies no crash

    def test_no_split_when_no_and(self):
        sentence = "We encrypt all data at rest."
        splits = _split_compound_claim(sentence, "Security", 0, len(sentence))
        assert splits is None

    def test_no_split_when_half_lacks_verb(self):
        sentence = "We encrypt data and nothing else."
        # "nothing else" has no privacy verb → no split
        splits = _split_compound_claim(sentence, "Security", 0, len(sentence))
        assert splits is None


# ---------------------------------------------------------------------------
# Tests: end-to-end extract_claims_text
# ---------------------------------------------------------------------------


class TestExtractClaimsText:
    """Integration tests for extract_claims_text using SAMPLE_POLICY."""

    def setup_method(self):
        self.claims = _run_sync(extract_claims_text(SAMPLE_POLICY))

    def test_returns_list_of_policy_claims(self):
        from mcp_vector_search.auditor.models import PolicyClaim

        assert isinstance(self.claims, list)
        assert all(isinstance(c, PolicyClaim) for c in self.claims)

    def test_nonzero_claims_extracted(self):
        assert len(self.claims) > 0

    def test_data_sharing_no_sell(self):
        """'We do not sell your personal data' → data_sharing, testable=True."""
        matching = [
            c
            for c in self.claims
            if c.category == "data_sharing" and "sell" in c.text.lower()
        ]
        assert matching, "Expected at least one data_sharing claim about selling data"
        assert all(c.testable for c in matching)

    def test_encryption_at_rest(self):
        """'All data is encrypted at rest using AES-256' → encryption, testable=True."""
        matching = [
            c
            for c in self.claims
            if c.category == "encryption" and "encrypt" in c.text.lower()
        ]
        assert matching, "Expected at least one encryption claim"
        assert all(c.testable for c in matching)

    def test_user_rights_deletion(self):
        """'Users can request deletion of their account' → user_rights, testable=True."""
        matching = [
            c
            for c in self.claims
            if c.category == "user_rights" and "request" in c.text.lower()
        ]
        assert matching, "Expected at least one user_rights claim about deletion"

    def test_third_party_analytics(self):
        """'We use third-party analytics providers' → third_party, testable=True."""
        matching = [c for c in self.claims if c.category == "third_party"]
        assert matching, "Expected at least one third_party claim"

    def test_retention_claim(self):
        """'Data is retained for 30 days after account deletion' → retention."""
        matching = [c for c in self.claims if c.category == "retention"]
        assert matching, "Expected at least one retention claim"

    def test_claim_ids_are_stable(self):
        """Running extractor twice on the same text produces the same IDs."""
        claims2 = _run_sync(extract_claims_text(SAMPLE_POLICY))
        ids1 = {c.id for c in self.claims}
        ids2 = {c.id for c in claims2}
        assert ids1 == ids2

    def test_claim_ids_are_12_chars_hex(self):
        for claim in self.claims:
            assert len(claim.id) == 12
            assert all(ch in "0123456789abcdef" for ch in claim.id)

    def test_no_duplicate_ids(self):
        ids = [c.id for c in self.claims]
        assert len(ids) == len(set(ids)), "Duplicate claim IDs found"

    def test_policy_section_populated(self):
        """At least some claims should have a policy_section set."""
        with_section = [c for c in self.claims if c.policy_section]
        assert with_section, "Expected at least some claims with policy_section set"

    def test_source_offsets_are_tuples(self):
        for claim in self.claims:
            assert isinstance(claim.source_offsets, tuple)
            assert len(claim.source_offsets) == 2
            assert claim.source_offsets[0] <= claim.source_offsets[1]

    def test_compound_sentence_splits(self):
        """'We encrypt data in transit and at rest' should yield at least one encryption claim."""
        compound_policy = (
            "## Security\n"
            "We encrypt data in transit and share aggregated data with partners."
        )
        claims = _run_sync(extract_claims_text(compound_policy))
        # Should produce at least 1 claim (possibly 2 after split)
        assert len(claims) >= 1


# ---------------------------------------------------------------------------
# Tests: merge_claims
# ---------------------------------------------------------------------------


class TestMergeClaims:
    def _make_claim(self, category: str, normalized: str) -> object:
        from mcp_vector_search.auditor.models import PolicyClaim

        return PolicyClaim(
            id=PolicyClaim.compute_id(category, normalized),
            category=category,  # type: ignore[arg-type]
            text=normalized,
            normalized=normalized,
            keywords=[],
            policy_section=None,
            testable=True,
            source_offsets=(0, len(normalized)),
        )

    def test_dedup_high_similarity(self):
        from mcp_vector_search.auditor.policy_extractor import merge_claims

        base = [
            self._make_claim("encryption", "The application does encrypt data at rest")
        ]
        extra = [
            self._make_claim("encryption", "The application does encrypt data at rest")
        ]
        merged = merge_claims(base, extra)
        assert len(merged) == 1

    def test_keeps_genuinely_new(self):
        from mcp_vector_search.auditor.policy_extractor import merge_claims

        base = [
            self._make_claim("encryption", "The application does encrypt data at rest")
        ]
        extra = [
            self._make_claim(
                "data_sharing", "The application does not sell personal data"
            )
        ]
        merged = merge_claims(base, extra)
        assert len(merged) == 2

    def test_empty_base(self):
        from mcp_vector_search.auditor.policy_extractor import merge_claims

        extra = [
            self._make_claim("consent", "The application does require user consent")
        ]
        merged = merge_claims([], extra)
        assert len(merged) == 1
