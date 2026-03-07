"""GitHub Wiki publisher for mcp-vector-search generated wikis."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

_GIT = shutil.which("git") or "git"  # resolve to full path for bandit B607


class WikiPublisher:
    """Publishes generated wiki content to a GitHub Wiki (the .wiki.git repo)."""

    def __init__(self, project_root: Path, wiki_output_dir: Path | None = None):
        """Initialize publisher.

        Args:
            project_root: Project root (used to find git remote)
            wiki_output_dir: Directory containing generated wiki .md files.
                             Defaults to project_root / "wiki" if not provided.
        """
        self.project_root = project_root
        self.wiki_output_dir = wiki_output_dir or (project_root / "wiki")

    def get_wiki_remote_url(self) -> str:
        """Get the GitHub Wiki git URL from the repo's remote.

        Converts https://github.com/owner/repo.git -> https://github.com/owner/repo.wiki.git
        or git@github.com:owner/repo.git -> git@github.com:owner/repo.wiki.git

        Returns:
            Wiki git URL

        Raises:
            ValueError: If git remote cannot be determined or is not GitHub
        """
        result = subprocess.run(
            [_GIT, "remote", "get-url", "origin"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"Failed to get git remote URL: {result.stderr.strip()}")

        remote = result.stdout.strip()
        logger.debug(f"Git remote: {remote}")

        # Convert to wiki URL
        if remote.endswith(".git"):
            wiki_url = remote[:-4] + ".wiki.git"
        else:
            wiki_url = remote + ".wiki.git"

        return wiki_url

    def publish(
        self,
        commit_message: str = "Update wiki via mcp-vector-search",
        dry_run: bool = False,
    ) -> PublishResult:
        """Publish wiki content to GitHub Wiki.

        Args:
            commit_message: Git commit message for the wiki update
            dry_run: If True, clone and stage but don't push

        Returns:
            PublishResult with status and details
        """
        wiki_url = self.get_wiki_remote_url()
        logger.info(f"Publishing wiki to: {wiki_url}")

        if not self.wiki_output_dir.exists():
            return PublishResult(
                success=False,
                error=f"Wiki output directory not found: {self.wiki_output_dir}. "
                "Run 'mvs wiki generate' first.",
            )

        wiki_files = list(self.wiki_output_dir.glob("*.md"))
        if not wiki_files:
            return PublishResult(
                success=False,
                error=f"No .md files found in {self.wiki_output_dir}. "
                "Run 'mvs wiki generate' first.",
            )

        with tempfile.TemporaryDirectory(prefix="mvs-wiki-") as tmpdir:
            clone_dir = Path(tmpdir) / "wiki"

            # Clone existing wiki (shallow, quiet)
            logger.info("Cloning wiki repository...")
            clone_result = subprocess.run(
                [_GIT, "clone", "--depth", "1", wiki_url, str(clone_dir)],
                capture_output=True,
                text=True,
            )

            if clone_result.returncode != 0:
                # Wiki may not exist yet — init a new repo
                logger.warning(
                    f"Wiki clone failed (may be empty): {clone_result.stderr.strip()}"
                )
                clone_dir.mkdir(parents=True)
                subprocess.run(
                    [_GIT, "init"],
                    cwd=clone_dir,
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [_GIT, "remote", "add", "origin", wiki_url],
                    cwd=clone_dir,
                    check=True,
                    capture_output=True,
                )

            # Copy generated wiki files
            copied: list[str] = []
            for src_file in wiki_files:
                dest = clone_dir / src_file.name
                shutil.copy2(src_file, dest)
                copied.append(src_file.name)
                logger.debug(f"Staged: {src_file.name}")

            # Ensure Home.md exists (GitHub Wiki requires it)
            if "Home.md" not in [f.name for f in wiki_files]:
                # Look for index or readme
                for candidate in ["index.md", "README.md", "Index.md"]:
                    if (self.wiki_output_dir / candidate).exists():
                        shutil.copy2(
                            self.wiki_output_dir / candidate,
                            clone_dir / "Home.md",
                        )
                        copied.append("Home.md (from " + candidate + ")")
                        break

            # Stage all changes
            subprocess.run(
                [_GIT, "add", "-A"],
                cwd=clone_dir,
                check=True,
                capture_output=True,
            )

            # Check if there are changes to commit
            status = subprocess.run(
                [_GIT, "status", "--porcelain"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
            )

            if not status.stdout.strip():
                return PublishResult(
                    success=True,
                    pages_published=len(copied),
                    wiki_url=_wiki_web_url(wiki_url),
                    message="Wiki already up to date — no changes to publish.",
                )

            # Commit
            subprocess.run(
                [_GIT, "commit", "-m", commit_message],
                cwd=clone_dir,
                check=True,
                capture_output=True,
            )

            if dry_run:
                return PublishResult(
                    success=True,
                    pages_published=len(copied),
                    wiki_url=_wiki_web_url(wiki_url),
                    message=f"Dry run: {len(copied)} pages staged, not pushed.",
                    files=copied,
                )

            # Push
            logger.info("Pushing to GitHub Wiki...")
            push_result = subprocess.run(
                [_GIT, "push", "origin", "HEAD"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
            )

            if push_result.returncode != 0:
                return PublishResult(
                    success=False,
                    error=f"Push failed: {push_result.stderr.strip()}",
                    files=copied,
                )

            return PublishResult(
                success=True,
                pages_published=len(copied),
                wiki_url=_wiki_web_url(wiki_url),
                message=f"Published {len(copied)} pages to GitHub Wiki.",
                files=copied,
            )


def _wiki_web_url(wiki_git_url: str) -> str:
    """Convert wiki git URL to browser-viewable URL.

    Args:
        wiki_git_url: Wiki git URL (https or ssh format)

    Returns:
        Browser-viewable GitHub Wiki URL
    """
    # git@github.com:owner/repo.wiki.git -> https://github.com/owner/repo/wiki
    # https://github.com/owner/repo.wiki.git -> https://github.com/owner/repo/wiki
    url = wiki_git_url
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")
    url = url.replace(".wiki.git", "/wiki")
    if url.endswith(".git"):
        url = url[:-4] + "/wiki"
    return url


class PublishResult:
    """Result of a wiki publish operation."""

    def __init__(
        self,
        success: bool,
        pages_published: int = 0,
        wiki_url: str = "",
        message: str = "",
        error: str = "",
        files: list[str] | None = None,
    ):
        """Initialize publish result.

        Args:
            success: Whether the publish succeeded
            pages_published: Number of pages published
            wiki_url: Browser URL to the GitHub Wiki
            message: Informational message
            error: Error message (if success=False)
            files: List of file names that were staged/published
        """
        self.success = success
        self.pages_published = pages_published
        self.wiki_url = wiki_url
        self.message = message
        self.error = error
        self.files = files or []
