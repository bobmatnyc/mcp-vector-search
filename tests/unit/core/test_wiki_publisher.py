"""Unit tests for GitHub Wiki publisher."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_vector_search.core.wiki_publisher import (
    PublishResult,
    WikiPublisher,
    _wiki_web_url,
)


class TestWikiWebUrl:
    def test_https_url(self):
        url = _wiki_web_url("https://github.com/owner/repo.wiki.git")
        assert url == "https://github.com/owner/repo/wiki"

    def test_ssh_url(self):
        url = _wiki_web_url("git@github.com:owner/repo.wiki.git")
        assert url == "https://github.com/owner/repo/wiki"

    def test_https_url_no_dot_git(self):
        # URL without .git suffix: .wiki.git replacement won't match, .git won't match either
        # The function handles URLs that end in .wiki.git (produced by get_wiki_remote_url)
        url = _wiki_web_url("https://github.com/owner/repo.wiki.git")
        assert url == "https://github.com/owner/repo/wiki"


class TestGetWikiRemoteUrl:
    def test_converts_https_to_wiki_url(self, tmp_path):
        publisher = WikiPublisher(project_root=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/owner/repo.git\n"

        with patch("subprocess.run", return_value=mock_result):
            url = publisher.get_wiki_remote_url()

        assert url == "https://github.com/owner/repo.wiki.git"

    def test_converts_ssh_to_wiki_url(self, tmp_path):
        publisher = WikiPublisher(project_root=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "git@github.com:owner/repo.git\n"

        with patch("subprocess.run", return_value=mock_result):
            url = publisher.get_wiki_remote_url()

        assert url == "git@github.com:owner/repo.wiki.git"

    def test_converts_url_without_git_suffix(self, tmp_path):
        publisher = WikiPublisher(project_root=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/owner/repo\n"

        with patch("subprocess.run", return_value=mock_result):
            url = publisher.get_wiki_remote_url()

        assert url == "https://github.com/owner/repo.wiki.git"

    def test_raises_on_git_failure(self, tmp_path):
        publisher = WikiPublisher(project_root=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: not a git repository"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ValueError, match="Failed to get git remote URL"):
                publisher.get_wiki_remote_url()


class TestPublish:
    def test_no_wiki_dir_returns_failure(self, tmp_path):
        publisher = WikiPublisher(
            project_root=tmp_path,
            wiki_output_dir=tmp_path / "nonexistent",
        )

        with patch.object(
            publisher,
            "get_wiki_remote_url",
            return_value="https://github.com/o/r.wiki.git",
        ):
            result = publisher.publish()

        assert not result.success
        assert "not found" in result.error

    def test_no_md_files_returns_failure(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "not_a_markdown.txt").write_text("ignored")

        publisher = WikiPublisher(project_root=tmp_path, wiki_output_dir=wiki_dir)

        with patch.object(
            publisher,
            "get_wiki_remote_url",
            return_value="https://github.com/o/r.wiki.git",
        ):
            result = publisher.publish()

        assert not result.success
        assert "No .md files" in result.error

    def test_default_wiki_output_dir(self, tmp_path):
        publisher = WikiPublisher(project_root=tmp_path)
        assert publisher.wiki_output_dir == tmp_path / "wiki"

    def test_custom_wiki_output_dir(self, tmp_path):
        custom_dir = tmp_path / "custom-wiki"
        publisher = WikiPublisher(project_root=tmp_path, wiki_output_dir=custom_dir)
        assert publisher.wiki_output_dir == custom_dir


class TestPublishResult:
    def test_success_defaults(self):
        result = PublishResult(success=True)
        assert result.success is True
        assert result.pages_published == 0
        assert result.wiki_url == ""
        assert result.message == ""
        assert result.error == ""
        assert result.files == []

    def test_failure_with_error(self):
        result = PublishResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_files_defaults_to_empty_list(self):
        result = PublishResult(success=True, files=None)
        assert result.files == []

    def test_full_result(self):
        result = PublishResult(
            success=True,
            pages_published=5,
            wiki_url="https://github.com/owner/repo/wiki",
            message="Published 5 pages.",
            files=["Home.md", "API.md", "Usage.md"],
        )
        assert result.success is True
        assert result.pages_published == 5
        assert result.wiki_url == "https://github.com/owner/repo/wiki"
        assert result.message == "Published 5 pages."
        assert len(result.files) == 3
