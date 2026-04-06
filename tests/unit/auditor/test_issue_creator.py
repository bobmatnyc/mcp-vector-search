"""Unit tests for the issue creator (M4 Task 15).

Tests cover:
- Issue creation for each reviewable status
- Skipping PASS and IGNORED verdicts
- Deduplication of existing issues
- Assignee / reviewer configuration
- Disabled via create_issues=False
- Missing token / missing repo configuration
- Title truncation for long claim texts
- Issue body contains evidence and cert reference
- Graceful handling of missing labels
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_vector_search.auditor.config import AuditorSettings
from mcp_vector_search.auditor.models import Evidence, Verdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    file_path: str = "src/api/client.py",
    start_line: int = 10,
    snippet: str = "session.verify = True",
) -> Evidence:
    return Evidence(
        tool="search_code",
        query="TLS encryption",
        file_path=file_path,
        start_line=start_line,
        end_line=start_line + 5,
        snippet=snippet,
        score=0.85,
    )


def _make_verdict(
    status: str = "FAIL",
    claim_id: str = "abc123def456",
    confidence: float = 0.75,
    reasoning: str = "The code contradicts the policy claim.",
    ignored: bool = False,
    evidence: list[Evidence] | None = None,
) -> Verdict:
    return Verdict(
        claim_id=claim_id,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence or [_make_evidence()],
        kg_path_present=False,
        evidence_count=1,
        ignored=ignored,
    )


def _make_settings(
    github_token: str = "ghp_testtoken",
    github_repo: str = "owner/myrepo",
    reviewer: str | None = None,
    create_issues: bool = True,
) -> AuditorSettings:
    """Build an AuditorSettings with GitHub config, bypassing env loading."""
    return AuditorSettings.model_construct(
        anthropic_api_key=None,
        extractor_model="claude-haiku-4-5",
        judge_model="claude-opus-4-6",
        min_evidence_count=2,
        require_kg_path=True,
        max_claims_per_policy=50,
        confidence_threshold=0.7,
        use_llm_extraction=True,
        openrouter_api_key=None,
        llm_backend="anthropic",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_extractor_model="anthropic/claude-haiku-4-5",
        openrouter_judge_model="anthropic/claude-sonnet-4-5",
        gpg_key_id=None,
        github_token=_secret(github_token) if github_token else None,
        github_repo=github_repo,
        reviewer=reviewer,
        create_issues=create_issues,
    )


def _secret(value: str):
    """Create a SecretStr-like object with get_secret_value()."""
    from pydantic import SecretStr

    return SecretStr(value)


def _mock_repo(existing_issues: list | None = None) -> MagicMock:
    """Build a mock PyGithub Repository object."""
    repo = MagicMock()
    repo.full_name = "owner/myrepo"

    # Labels
    label1 = MagicMock()
    label1.name = "privacy-audit"
    repo.get_labels.return_value = [label1]
    repo.create_label.return_value = MagicMock()

    # Existing issues for dedup
    repo.get_issues.return_value = iter(existing_issues or [])

    # New issue creation
    new_issue = MagicMock()
    new_issue.number = 42
    new_issue.html_url = "https://github.com/owner/myrepo/issues/42"
    repo.create_issue.return_value = new_issue

    return repo


def _mock_github_cls(repo: MagicMock) -> MagicMock:
    """Return a mock Github class that yields the given repo."""
    gh_instance = MagicMock()
    gh_instance.get_repo.return_value = repo
    gh_cls = MagicMock(return_value=gh_instance)
    return gh_cls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_issues_for_fail_verdict():
    """FAIL verdicts create issues with privacy-fail label and correct title."""
    verdict = _make_verdict(status="FAIL", claim_id="abc123def456")
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123def456abcdef",
            settings=settings,
        )

    assert len(results) == 1
    assert results[0]["status"] == "created"
    assert results[0]["claim_id"] == "abc123def456"

    # Verify labels passed to create_issue include privacy-fail
    call_kwargs = repo.create_issue.call_args[1]
    assert "privacy-fail" in call_kwargs["labels"]
    assert "privacy-audit" in call_kwargs["labels"]

    # Title must start with the right prefix
    assert "[Privacy Audit] FAIL:" in call_kwargs["title"]


@pytest.mark.asyncio
async def test_create_issues_for_manual_review():
    """MANUAL_REVIEW verdicts create issues with privacy-review label."""
    verdict = _make_verdict(status="MANUAL_REVIEW", claim_id="rev111aabbcc")
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="deadbeef1234",
            settings=settings,
        )

    assert len(results) == 1
    call_kwargs = repo.create_issue.call_args[1]
    assert "privacy-review" in call_kwargs["labels"]
    assert "MANUAL_REVIEW" in call_kwargs["title"]


@pytest.mark.asyncio
async def test_create_issues_for_insufficient_evidence():
    """INSUFFICIENT_EVIDENCE verdicts create issues with privacy-insufficient label."""
    verdict = _make_verdict(status="INSUFFICIENT_EVIDENCE", claim_id="insuf9988776")
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="cafebabe1234",
            settings=settings,
        )

    assert len(results) == 1
    call_kwargs = repo.create_issue.call_args[1]
    assert "privacy-insufficient" in call_kwargs["labels"]
    assert "INSUFFICIENT_EVIDENCE" in call_kwargs["title"]


@pytest.mark.asyncio
async def test_skip_pass_verdicts():
    """PASS verdicts must not create any GitHub issues."""
    pass_verdict = _make_verdict(status="PASS", claim_id="pass001aabb")
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[pass_verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert results == []
    repo.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_skip_ignored_verdicts():
    """Ignored verdicts (ignored=True) must not create any GitHub issues."""
    ignored_verdict = _make_verdict(
        status="MANUAL_REVIEW", claim_id="ign001aabbcc", ignored=True
    )
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[ignored_verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert results == []
    repo.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_dedupe_existing_issue():
    """When an open issue with the claim_id already exists, add a comment instead of creating."""
    claim_id = "dup001aabbcc"
    verdict = _make_verdict(status="FAIL", claim_id=claim_id)
    settings = _make_settings()

    # Build a fake existing issue that contains the claim_id in its body
    existing = MagicMock()
    existing.number = 99
    existing.html_url = "https://github.com/owner/myrepo/issues/99"
    existing.body = f"Some body text with claim_id {claim_id} embedded"
    existing.create_comment = MagicMock()

    repo = _mock_repo(existing_issues=[existing])
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert len(results) == 1
    assert results[0]["status"] == "updated"
    assert results[0]["issue_number"] == 99
    # A comment must have been added, not a new issue
    existing.create_comment.assert_called_once()
    repo.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_assignee_set_when_reviewer_configured():
    """When reviewer is configured, the assignee is set on the created issue."""
    verdict = _make_verdict(status="FAIL", claim_id="rev001xxyyzz")
    settings = _make_settings(reviewer="octocat")
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert len(results) == 1
    call_kwargs = repo.create_issue.call_args[1]
    assert call_kwargs.get("assignees") == ["octocat"]


@pytest.mark.asyncio
async def test_no_issues_when_disabled():
    """Setting create_issues=False must skip all issue creation."""
    verdict = _make_verdict(status="FAIL", claim_id="dis001xxyyzz")
    settings = _make_settings(create_issues=False)
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert results == []
    repo.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_no_issues_without_token():
    """Missing github_token logs a warning and returns empty list."""
    verdict = _make_verdict(status="FAIL", claim_id="notok001xxyy")
    settings = _make_settings(github_token="")
    # Manually set github_token to None to simulate missing token
    settings = settings.model_copy(update={"github_token": None})

    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert results == []
    repo.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_no_issues_without_repo():
    """Missing github_repo logs a warning and returns empty list."""
    verdict = _make_verdict(status="FAIL", claim_id="norepo01xxyy")
    settings = _make_settings(github_repo="")
    # Remove repo info so parsing fails
    settings = settings.model_copy(update={"github_repo": None})

    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            # Pass a bare filesystem path that can't be parsed as owner/repo
            target_repo="/home/user/projects/myproject",
            target_commit="abc123",
            settings=settings,
        )

    assert results == []
    repo.create_issue.assert_not_called()


@pytest.mark.asyncio
async def test_issue_title_truncation():
    """Claim text longer than 60 chars is truncated with '...' in the title."""
    long_reasoning = "A" * 100 + " some extra words that push it way over the limit"
    verdict = _make_verdict(
        status="FAIL",
        claim_id="trunc01xxyyz",
        reasoning=long_reasoning,
    )
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    assert len(results) == 1
    title = repo.create_issue.call_args[1]["title"]
    # After "[Privacy Audit] FAIL: " prefix, the claim portion must end with "..."
    claim_part = title.split("[Privacy Audit] FAIL: ", 1)[1]
    assert claim_part.endswith("...")
    # The claim portion must be at most 60 + len("...") = 63 chars
    assert len(claim_part) <= 63


@pytest.mark.asyncio
async def test_issue_body_contains_evidence():
    """Issue body contains up to 3 evidence snippets with file:line references."""
    evidences = [
        _make_evidence(file_path="src/a.py", start_line=1, snippet="code_a"),
        _make_evidence(file_path="src/b.py", start_line=2, snippet="code_b"),
        _make_evidence(file_path="src/c.py", start_line=3, snippet="code_c"),
        _make_evidence(
            file_path="src/d.py", start_line=4, snippet="code_d"
        ),  # 4th — should not appear
    ]
    verdict = _make_verdict(status="FAIL", claim_id="evid01xxyyzz", evidence=evidences)
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    body = repo.create_issue.call_args[1]["body"]
    assert "src/a.py" in body
    assert "src/b.py" in body
    assert "src/c.py" in body
    assert "src/d.py" not in body  # 4th evidence must not appear


@pytest.mark.asyncio
async def test_issue_body_contains_cert_reference():
    """Issue body contains the certification path reference."""
    verdict = _make_verdict(status="MANUAL_REVIEW", claim_id="cert01xxyyzz")
    settings = _make_settings()
    repo = _mock_repo()
    gh_cls = _mock_github_cls(repo)
    cert_path = Path("certifications/myrepo/20260101-120000/certification.md")

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
            cert_path=cert_path,
        )

    body = repo.create_issue.call_args[1]["body"]
    assert "certifications/myrepo/20260101-120000/certification.md" in body


@pytest.mark.asyncio
async def test_labels_created_if_missing():
    """Labels that don't exist on the repo are created gracefully."""
    verdict = _make_verdict(status="FAIL", claim_id="lbl001xxyyzz")
    settings = _make_settings()

    repo = MagicMock()
    repo.full_name = "owner/myrepo"
    repo.get_labels.return_value = []  # No existing labels
    repo.create_label.return_value = MagicMock()
    repo.get_issues.return_value = iter([])

    new_issue = MagicMock()
    new_issue.number = 77
    new_issue.html_url = "https://github.com/owner/myrepo/issues/77"
    repo.create_issue.return_value = new_issue

    gh_cls = _mock_github_cls(repo)

    with (
        patch("mcp_vector_search.auditor.issue_creator.Github", gh_cls),
        patch("mcp_vector_search.auditor.issue_creator._PYGITHUB_AVAILABLE", True),
    ):
        from mcp_vector_search.auditor.issue_creator import create_review_issues

        results = await create_review_issues(
            verdicts=[verdict],
            target_repo="owner/myrepo",
            target_commit="abc123",
            settings=settings,
        )

    # create_label must have been called for the missing labels
    assert repo.create_label.call_count >= 1
    created_names = {call[1]["name"] for call in repo.create_label.call_args_list}
    assert "privacy-audit" in created_names or "privacy-fail" in created_names

    # Issue still created despite label creation
    assert len(results) == 1
    assert results[0]["status"] == "created"
