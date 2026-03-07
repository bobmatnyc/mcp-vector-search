"""Unit tests for on-demand git blame enrichment."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vector_search.core.git_blame import _parse_porcelain, enrich_with_git_blame
from mcp_vector_search.core.models import SearchResult

SAMPLE_PORCELAIN = """\
abc123def456abc123def456abc123def456abc1 1 1 3
author Jane Developer
author-mail <jane@example.com>
author-time 1709500000
author-tz +0000
committer Jane Developer
committer-mail <jane@example.com>
committer-time 1709500000
committer-tz +0000
summary Fix authentication bug
filename auth/service.py
\tdef authenticate(user, password):
"""


def make_result(**kwargs) -> SearchResult:
    defaults = {
        "content": "def authenticate(user, password): pass",
        "file_path": Path("/repo/auth/service.py"),
        "start_line": 1,
        "end_line": 3,
        "language": "python",
        "similarity_score": 0.8,
        "rank": 1,
    }
    defaults.update(kwargs)
    return SearchResult(**defaults)


class TestParsePorcelain:
    def test_parses_author_name_and_email(self):
        result = _parse_porcelain(SAMPLE_PORCELAIN)
        assert result is not None
        assert "Jane Developer" in result["author"]
        assert "jane@example.com" in result["author"]

    def test_parses_commit_hash(self):
        result = _parse_porcelain(SAMPLE_PORCELAIN)
        assert result["commit"].startswith("abc123")

    def test_parses_timestamp_to_iso_date(self):
        result = _parse_porcelain(SAMPLE_PORCELAIN)
        assert result["author-time-iso"] == "2024-03-03"

    def test_empty_output_returns_none(self):
        assert _parse_porcelain("") is None

    def test_malformed_output_returns_none(self):
        assert _parse_porcelain("not porcelain output\n") is None


class TestEnrichWithGitBlame:
    @pytest.mark.asyncio
    async def test_populates_author_fields(self):
        result = make_result()
        assert result.last_author is None

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(SAMPLE_PORCELAIN.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await enrich_with_git_blame([result], Path("/repo"))

        assert result.last_author is not None
        assert "Jane Developer" in result.last_author
        assert result.commit_hash is not None
        assert result.last_modified == "2024-03-03"

    @pytest.mark.asyncio
    async def test_non_fatal_on_git_blame_failure(self):
        result = make_result()

        mock_proc = MagicMock()
        mock_proc.returncode = 128  # git error
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repo"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Should not raise
            await enrich_with_git_blame([result], Path("/repo"))

        assert result.last_author is None

    @pytest.mark.asyncio
    async def test_non_fatal_on_subprocess_exception(self):
        result = make_result()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("git not found"),
        ):
            # Should not raise
            await enrich_with_git_blame([result], Path("/repo"))

        assert result.last_author is None

    @pytest.mark.asyncio
    async def test_empty_results_list(self):
        # Should handle empty list without error
        await enrich_with_git_blame([], Path("/repo"))
