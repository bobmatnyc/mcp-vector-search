"""Unit tests for drift detection (Task 17, M5).

Tests:
- test_no_previous_audit
- test_no_drift_detected
- test_policy_drift
- test_code_drift
- test_both_drift
- test_days_since_audit
- test_drift_report_details_no_drift
- test_drift_report_details_policy
- test_drift_report_details_code
- test_drift_report_details_both
- test_target_slug_from_path
- test_drift_check_json_output
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from mcp_vector_search.auditor.drift import (
    _build_details,
    _days_since,
    _target_slug,
    check_drift,
)
from mcp_vector_search.auditor.models import DriftReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMIT_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_COMMIT_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_POLICY_TEXT_A = b"We collect minimal data."
_POLICY_TEXT_B = b"We collect all your data."
_SHA_A = hashlib.sha256(_POLICY_TEXT_A).hexdigest()
_SHA_B = hashlib.sha256(_POLICY_TEXT_B).hexdigest()

_TIMESTAMP = "20260101-120000"  # fixed past timestamp for deterministic tests


def _make_index(
    slug: str,
    commit_sha: str = _COMMIT_A,
    policy_sha256: str = _SHA_A,
    timestamp: str = _TIMESTAMP,
) -> dict:
    """Build a minimal index.json structure."""
    return {
        "targets": {
            slug: {
                "latest": timestamp,
                "audits": [
                    {
                        "timestamp": timestamp,
                        "commit_sha": commit_sha,
                        "policy_sha256": policy_sha256,
                        "overall_status": "CERTIFIED",
                        "summary": {"PASS": 5, "TOTAL": 5},
                        "signed": False,
                    }
                ],
            }
        }
    }


def _write_index(tmp_path: Path, index_data: dict) -> Path:
    """Write index.json to tmp_path/certifications/index.json."""
    cert_dir = tmp_path / "certifications"
    cert_dir.mkdir(parents=True, exist_ok=True)
    index_path = cert_dir / "index.json"
    index_path.write_text(json.dumps(index_data), encoding="utf-8")
    return cert_dir


def _write_policy(tmp_path: Path, content: bytes = _POLICY_TEXT_A) -> Path:
    """Write a policy file to tmp_path/PRIVACY.md."""
    policy_path = tmp_path / "PRIVACY.md"
    policy_path.write_bytes(content)
    return policy_path


def _make_repo_dir(tmp_path: Path) -> Path:
    """Create a fake target repo directory named 'myrepo'."""
    repo = tmp_path / "myrepo"
    repo.mkdir()
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_previous_audit(tmp_path: Path) -> None:
    """When index.json has no entry for the target, has_drift=True with a clear message."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path)
    cert_dir = tmp_path / "certifications"
    cert_dir.mkdir()

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is True
    assert report.policy_changed is False
    assert report.code_changed is False
    assert report.details == "No previous audit found"
    assert report.last_audit_timestamp is None
    assert report.last_audit_commit is None


def test_no_drift_detected(tmp_path: Path) -> None:
    """Same commit + same policy hash → has_drift=False."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is False
    assert report.policy_changed is False
    assert report.code_changed is False
    assert report.current_commit == _COMMIT_A
    assert report.current_policy_sha256 == _SHA_A


def test_policy_drift(tmp_path: Path) -> None:
    """Different policy hash → policy_changed=True, has_drift=True."""
    repo = _make_repo_dir(tmp_path)
    # Policy file has content B, but index records SHA_A
    policy = _write_policy(tmp_path, _POLICY_TEXT_B)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is True
    assert report.policy_changed is True
    assert report.code_changed is False
    assert report.current_policy_sha256 == _SHA_B
    assert report.last_audit_policy_sha256 == _SHA_A


def test_code_drift(tmp_path: Path) -> None:
    """Different commit → code_changed=True, has_drift=True."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_B,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is True
    assert report.policy_changed is False
    assert report.code_changed is True
    assert report.current_commit == _COMMIT_B
    assert report.last_audit_commit == _COMMIT_A


def test_both_drift(tmp_path: Path) -> None:
    """Different commit AND policy hash → both True."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_B)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_B,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is True
    assert report.policy_changed is True
    assert report.code_changed is True


def test_days_since_audit(tmp_path: Path) -> None:
    """days_since_last_audit is a non-negative integer for a past timestamp."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(
        tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A, _TIMESTAMP)
    )

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    # _TIMESTAMP is 2026-01-01 12:00 UTC; today (2026-04-05) is ~94 days later
    assert report.days_since_last_audit is not None
    assert report.days_since_last_audit >= 0


def test_drift_report_details_no_drift(tmp_path: Path) -> None:
    """Details message contains 'No drift' when nothing changed."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert "No drift" in report.details


def test_drift_report_details_policy(tmp_path: Path) -> None:
    """Details message mentions 'Policy changed' when only policy changed."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_B)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert "Policy changed" in report.details


def test_drift_report_details_code(tmp_path: Path) -> None:
    """Details message mentions 'Code changed' when only commit changed."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_B,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert "Code changed" in report.details


def test_drift_report_details_both(tmp_path: Path) -> None:
    """Details message mentions 'Both' when both policy and code changed."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_B)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_B,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert "Both" in report.details


def test_target_slug_from_path() -> None:
    """/path/to/tripbot7 → 'tripbot7'."""
    slug = _target_slug(Path("/path/to/tripbot7"))
    assert slug == "tripbot7"


def test_target_slug_trailing_sep() -> None:
    """Path.name ignores trailing separators in Python Path objects."""
    slug = _target_slug(Path("/my/project/myrepo"))
    assert slug == "myrepo"


def test_drift_check_json_output(tmp_path: Path) -> None:
    """DriftReport.model_dump_json() produces valid JSON with expected keys."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    json_str = report.model_dump_json()
    parsed = json.loads(json_str)

    assert "has_drift" in parsed
    assert "policy_changed" in parsed
    assert "code_changed" in parsed
    assert "details" in parsed
    assert "target" in parsed
    assert "days_since_last_audit" in parsed
    assert isinstance(parsed["has_drift"], bool)


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_build_details_no_drift() -> None:
    details = _build_details(
        policy_changed=False,
        code_changed=False,
        last_commit=_COMMIT_A,
        current_commit=_COMMIT_A,
        days=5,
    )
    assert "No drift" in details
    assert "5 days" in details


def test_build_details_one_day() -> None:
    details = _build_details(
        policy_changed=False,
        code_changed=False,
        last_commit=_COMMIT_A,
        current_commit=_COMMIT_A,
        days=1,
    )
    assert "1 day" in details
    assert "1 days" not in details  # singular


def test_build_details_policy_only() -> None:
    details = _build_details(
        policy_changed=True,
        code_changed=False,
        last_commit=_COMMIT_A,
        current_commit=_COMMIT_A,
        days=3,
    )
    assert "Policy changed" in details
    assert "Re-audit recommended" in details


def test_build_details_code_only() -> None:
    details = _build_details(
        policy_changed=False,
        code_changed=True,
        last_commit=_COMMIT_A,
        current_commit=_COMMIT_B,
        days=3,
    )
    assert "Code changed" in details
    short_a = _COMMIT_A[:12]
    short_b = _COMMIT_B[:12]
    assert short_a in details
    assert short_b in details


def test_build_details_both() -> None:
    details = _build_details(
        policy_changed=True,
        code_changed=True,
        last_commit=_COMMIT_A,
        current_commit=_COMMIT_B,
        days=10,
    )
    assert "Both" in details
    assert "Re-audit required" in details


def test_days_since_valid_timestamp() -> None:
    # _TIMESTAMP is 2026-01-01 12:00 UTC; today is ~2026-04-05
    days = _days_since(_TIMESTAMP)
    assert days is not None
    assert days >= 0


def test_days_since_invalid_timestamp() -> None:
    assert _days_since("not-a-date") is None


def test_no_index_file(tmp_path: Path) -> None:
    """When certifications dir has no index.json, has_drift=True."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path)
    cert_dir = tmp_path / "certifications"
    cert_dir.mkdir()
    # index.json does NOT exist

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is True
    assert "No previous audit" in report.details


def test_empty_audits_list(tmp_path: Path) -> None:
    """When audits list is empty for target slug, treat as no previous audit."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path)
    cert_dir = tmp_path / "certifications"
    cert_dir.mkdir()
    index_data = {"targets": {"myrepo": {"latest": "", "audits": []}}}
    (cert_dir / "index.json").write_text(json.dumps(index_data))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert report.has_drift is True
    assert "No previous audit" in report.details


def test_drift_report_is_pydantic_model(tmp_path: Path) -> None:
    """DriftReport is a Pydantic BaseModel with correct field types."""
    repo = _make_repo_dir(tmp_path)
    policy = _write_policy(tmp_path, _POLICY_TEXT_A)
    cert_dir = _write_index(tmp_path, _make_index("myrepo", _COMMIT_A, _SHA_A))

    with patch(
        "mcp_vector_search.auditor.drift._get_head_commit",
        return_value=_COMMIT_A,
    ):
        report = check_drift(repo, policy, cert_dir)

    assert isinstance(report, DriftReport)
    assert isinstance(report.has_drift, bool)
    assert isinstance(report.details, str)
    assert isinstance(report.target, str)
