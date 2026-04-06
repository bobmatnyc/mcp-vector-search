"""Unit tests for the ignore list (Task 3).

Tests:
- Loading from a valid YAML file
- Rejecting entries with justification < 20 chars
- Warning on expired entries (not including them)
- IgnoreList.matches() by claim_id, category, and pattern
- Empty ignore file / no ignore file
"""

from pathlib import Path

import yaml

from mcp_vector_search.auditor.ignore import IgnoreList
from mcp_vector_search.auditor.models import PolicyClaim


def _make_claim(
    claim_id: str = "abc123def456",
    category: str = "encryption",
    normalized: str = "All API calls use TLS 1.2.",
) -> PolicyClaim:
    return PolicyClaim(
        id=claim_id,
        category=category,
        text="We use TLS for all communications.",
        normalized=normalized,
        keywords=["TLS"],
        policy_section="Security",
        testable=True,
        source_offsets=(0, 100),
    )


def _write_ignore_file(tmp_path: Path, entries: list[dict]) -> Path:
    ignore_file = tmp_path / ".audit-ignore.yml"
    ignore_file.write_text(yaml.dump(entries), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestIgnoreListLoad:
    def test_no_file_returns_empty(self, tmp_path):
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 0

    def test_valid_file_loads(self, tmp_path):
        _write_ignore_file(
            tmp_path,
            [
                {
                    "claim_id": "abc123def456",
                    "justification": "Manually verified this claim in production environment.",
                    "approved_by": "alice",
                }
            ],
        )
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 1

    def test_short_justification_is_rejected(self, tmp_path):
        _write_ignore_file(
            tmp_path,
            [
                {
                    "claim_id": "abc123def456",
                    "justification": "Too short",  # < 20 chars
                    "approved_by": "alice",
                }
            ],
        )
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 0

    def test_expired_entry_is_excluded(self, tmp_path):
        _write_ignore_file(
            tmp_path,
            [
                {
                    "claim_id": "abc123def456",
                    "justification": "This was valid until the old audit cycle.",
                    "approved_by": "alice",
                    "expires": "2020-01-01",  # past date
                }
            ],
        )
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 0

    def test_future_expiry_is_included(self, tmp_path):
        _write_ignore_file(
            tmp_path,
            [
                {
                    "claim_id": "abc123def456",
                    "justification": "Valid until the end of current audit cycle.",
                    "approved_by": "alice",
                    "expires": "2099-12-31",
                }
            ],
        )
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 1

    def test_invalid_yaml_returns_empty(self, tmp_path):
        ignore_file = tmp_path / ".audit-ignore.yml"
        ignore_file.write_text("{{{{invalid yaml", encoding="utf-8")
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 0

    def test_non_list_yaml_returns_empty(self, tmp_path):
        ignore_file = tmp_path / ".audit-ignore.yml"
        ignore_file.write_text("key: value\n", encoding="utf-8")
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 0

    def test_non_dict_item_is_skipped(self, tmp_path):
        ignore_file = tmp_path / ".audit-ignore.yml"
        ignore_file.write_text(
            "- just a string\n- claim_id: abc\n  justification: long enough justification here\n  approved_by: alice\n",
            encoding="utf-8",
        )
        ignore_list = IgnoreList.load(tmp_path)
        # Only the second item (which happens to be malformed re: justification length)
        # First item "just a string" is not a dict — skipped
        # The dict item has valid justification >= 20 chars
        assert len(ignore_list) == 1

    def test_mixed_valid_and_invalid_entries(self, tmp_path):
        _write_ignore_file(
            tmp_path,
            [
                {
                    "claim_id": "id1",
                    "justification": "Valid justification that is long enough",
                    "approved_by": "alice",
                },
                {
                    "claim_id": "id2",
                    "justification": "short",  # rejected
                    "approved_by": "bob",
                },
            ],
        )
        ignore_list = IgnoreList.load(tmp_path)
        assert len(ignore_list) == 1


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestIgnoreListMatches:
    def _make_ignore_list(self, entries: list[dict]) -> IgnoreList:
        from mcp_vector_search.auditor.models import AuditIgnoreEntry

        validated = []
        for e in entries:
            try:
                validated.append(AuditIgnoreEntry.model_validate(e))
            except Exception:
                pass
        return IgnoreList(validated)

    def test_no_match_returns_none(self):
        ignore_list = self._make_ignore_list([])
        claim = _make_claim()
        assert ignore_list.matches(claim) is None

    def test_claim_id_match(self):
        ignore_list = self._make_ignore_list(
            [
                {
                    "claim_id": "abc123def456",
                    "justification": "Verified manually during security review.",
                    "approved_by": "alice",
                }
            ]
        )
        claim = _make_claim(claim_id="abc123def456")
        result = ignore_list.matches(claim)
        assert result is not None
        assert result.claim_id == "abc123def456"

    def test_claim_id_no_match(self):
        ignore_list = self._make_ignore_list(
            [
                {
                    "claim_id": "different000",
                    "justification": "Verified manually during security review.",
                    "approved_by": "alice",
                }
            ]
        )
        claim = _make_claim(claim_id="abc123def456")
        assert ignore_list.matches(claim) is None

    def test_category_match(self):
        ignore_list = self._make_ignore_list(
            [
                {
                    "category": "encryption",
                    "justification": "Encryption handled at infrastructure layer by ops team.",
                    "approved_by": "security-team",
                }
            ]
        )
        claim = _make_claim(category="encryption")
        result = ignore_list.matches(claim)
        assert result is not None
        assert result.category == "encryption"

    def test_category_no_match(self):
        ignore_list = self._make_ignore_list(
            [
                {
                    "category": "retention",
                    "justification": "Retention handled at infrastructure layer by ops team.",
                    "approved_by": "security-team",
                }
            ]
        )
        claim = _make_claim(category="encryption")
        assert ignore_list.matches(claim) is None

    def test_pattern_match(self):
        ignore_list = self._make_ignore_list(
            [
                {
                    "pattern": "TLS.*1\\.2",
                    "justification": "TLS 1.2 requirement covered by infrastructure team.",
                    "approved_by": "alice",
                }
            ]
        )
        claim = _make_claim(normalized="All API calls use TLS 1.2.")
        result = ignore_list.matches(claim)
        assert result is not None

    def test_pattern_no_match(self):
        ignore_list = self._make_ignore_list(
            [
                {
                    "pattern": "bcrypt|argon2",
                    "justification": "Password hashing is handled at infra level by ops.",
                    "approved_by": "alice",
                }
            ]
        )
        claim = _make_claim(normalized="All API calls use TLS 1.2.")
        assert ignore_list.matches(claim) is None

    def test_claim_id_takes_priority_over_category(self):
        """claim_id match is checked before category match."""
        from mcp_vector_search.auditor.models import AuditIgnoreEntry

        entry_by_id = AuditIgnoreEntry(
            claim_id="abc123def456",
            justification="Matched by ID — approved in last audit cycle.",
            approved_by="alice",
        )
        entry_by_cat = AuditIgnoreEntry(
            category="encryption",
            justification="Matched by category — infra team handles.",
            approved_by="bob",
        )
        ignore_list = IgnoreList([entry_by_id, entry_by_cat])
        claim = _make_claim(claim_id="abc123def456", category="encryption")
        result = ignore_list.matches(claim)
        assert result is entry_by_id
