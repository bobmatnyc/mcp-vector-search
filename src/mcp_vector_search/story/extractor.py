"""Git and GitHub data extraction for story generation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    CommitInfo,
    ContributorInfo,
    DocReference,
    IssueInfo,
    PullRequestInfo,
    StoryExtraction,
)

logger = logging.getLogger(__name__)


class StoryExtractor:
    """Extracts git history and GitHub metadata for story generation.

    Responsibilities:
    - Parse git log for commits and contributors
    - Extract GitHub issues and PRs via gh CLI
    - Scan for documentation files
    - Filter noise (merge commits, dependency bumps)
    - Run extractions concurrently for performance
    """

    def __init__(
        self,
        project_root: Path,
        include_merge_commits: bool = False,
        filter_dep_bumps: bool = True,
    ) -> None:
        """Initialize extractor.

        Args:
            project_root: Path to git repository root
            include_merge_commits: Whether to include merge commits
            filter_dep_bumps: Whether to filter dependency update commits
        """
        self.project_root = project_root
        self.include_merge_commits = include_merge_commits
        self.filter_dep_bumps = filter_dep_bumps

    async def extract_all(self) -> StoryExtraction:
        """Extract all available data concurrently.

        Returns:
            StoryExtraction with commits, contributors, issues, PRs, and docs
        """
        logger.info("Starting story data extraction from %s", self.project_root)

        # Run all extractions concurrently
        results = await asyncio.gather(
            self._extract_commits(),
            self._extract_contributors(),
            self._extract_issues(),
            self._extract_pull_requests(),
            self._extract_docs(),
            return_exceptions=True,
        )

        # Unpack results and handle errors
        commits = results[0] if not isinstance(results[0], Exception) else []
        contributors = results[1] if not isinstance(results[1], Exception) else []
        issues = results[2] if not isinstance(results[2], Exception) else []
        pull_requests = results[3] if not isinstance(results[3], Exception) else []
        docs = results[4] if not isinstance(results[4], Exception) else []

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                section = [
                    "commits",
                    "contributors",
                    "issues",
                    "pull_requests",
                    "docs",
                ][i]
                logger.warning("Failed to extract %s: %s", section, result)

        logger.info(
            "Extraction complete: %d commits, %d contributors, %d issues, %d PRs, %d docs",
            len(commits),
            len(contributors),
            len(issues),
            len(pull_requests),
            len(docs),
        )

        return StoryExtraction(
            commits=commits,
            contributors=contributors,
            issues=issues,
            pull_requests=pull_requests,
            docs=docs,
        )

    async def _extract_commits(self) -> list[CommitInfo]:
        """Extract commit history from git log.

        Format: git log --format=%H|%h|%s|%b|%an|%ae|%aI --numstat
        Each commit has:
        - Header line: hash|short_hash|subject|body|author_name|email|date
        - Followed by numstat lines: insertions  deletions  filename

        Returns:
            List of CommitInfo objects
        """
        logger.debug("Extracting commits from git log")

        # Build git log command
        format_str = "%H|%h|%s|%b|%an|%ae|%aI"
        cmd = [
            "git",
            "log",
            f"--format={format_str}",
            "--numstat",
        ]

        if not self.include_merge_commits:
            cmd.append("--no-merges")

        # Execute git log
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"git log failed: {error_msg}")

        # Parse output
        output = stdout.decode("utf-8", errors="replace")
        commits = self._parse_git_log(output)

        # Filter dependency bumps if requested
        if self.filter_dep_bumps:
            commits = self._filter_dependency_commits(commits)

        logger.debug("Extracted %d commits", len(commits))
        return commits

    def _parse_git_log(self, output: str) -> list[CommitInfo]:
        """Parse git log output into CommitInfo objects.

        Args:
            output: Raw git log output

        Returns:
            List of parsed CommitInfo objects
        """
        commits: list[CommitInfo] = []
        lines = output.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Check if this is a commit header (has pipes)
            if "|" not in line:
                i += 1
                continue

            # Parse commit header
            parts = line.split("|", 6)
            if len(parts) < 7:
                i += 1
                continue

            (
                commit_hash,
                short_hash,
                subject,
                body,
                author_name,
                author_email,
                date_str,
            ) = parts

            # Parse date
            try:
                commit_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning("Failed to parse date: %s", date_str)
                i += 1
                continue

            # Collect numstat lines (insertions/deletions/filename)
            files: list[str] = []
            total_insertions = 0
            total_deletions = 0
            i += 1

            while i < len(lines):
                stat_line = lines[i]

                # Empty line or next commit header means we're done with this commit
                if not stat_line.strip() or "|" in stat_line:
                    break

                # Parse numstat: insertions deletions filename
                stat_parts = stat_line.split("\t")
                if len(stat_parts) >= 3:
                    ins_str, del_str, filename = (
                        stat_parts[0],
                        stat_parts[1],
                        stat_parts[2],
                    )

                    # Handle binary files (marked with -)
                    insertions = 0 if ins_str == "-" else int(ins_str)
                    deletions = 0 if del_str == "-" else int(del_str)

                    total_insertions += insertions
                    total_deletions += deletions
                    files.append(filename)

                i += 1

            # Check if merge commit
            is_merge = "Merge" in subject or "merge" in subject.lower()

            commits.append(
                CommitInfo(
                    hash=commit_hash,
                    short_hash=short_hash,
                    message=subject,
                    body=body,
                    author_name=author_name,
                    author_email=author_email,
                    date=commit_date,
                    files_changed=len(files),
                    insertions=total_insertions,
                    deletions=total_deletions,
                    is_merge=is_merge,
                    files=files,
                )
            )

        return commits

    def _filter_dependency_commits(self, commits: list[CommitInfo]) -> list[CommitInfo]:
        """Filter out dependency update commits.

        Looks for common patterns:
        - "bump", "update", "upgrade" with package names
        - "deps:", "chore(deps):"
        - package.json, requirements.txt, Cargo.toml changes

        Args:
            commits: List of commits to filter

        Returns:
            Filtered list of commits
        """
        dep_patterns = [
            r"bump\s+\S+\s+(from|to|version)",
            r"update\s+(dependencies|deps|packages)",
            r"upgrade\s+\S+\s+(from|to)",
            r"deps?:|chore\(deps?\):",
            r"^chore:\s*(bump|update|upgrade)",
        ]

        dep_files = {
            "package.json",
            "package-lock.json",
            "requirements.txt",
            "Pipfile.lock",
            "Cargo.lock",
            "go.mod",
            "go.sum",
            "yarn.lock",
            "pnpm-lock.yaml",
        }

        filtered: list[CommitInfo] = []

        for commit in commits:
            message_lower = commit.message.lower()

            # Check message patterns
            is_dep_commit = any(
                re.search(pattern, message_lower) for pattern in dep_patterns
            )

            # Check if only dependency files changed
            if not is_dep_commit and commit.files:
                file_names = {Path(f).name for f in commit.files}
                is_dep_commit = bool(file_names and file_names.issubset(dep_files))

            if not is_dep_commit:
                filtered.append(commit)

        logger.debug("Filtered %d dependency commits", len(commits) - len(filtered))
        return filtered

    async def _extract_contributors(self) -> list[ContributorInfo]:
        """Extract contributor statistics from git shortlog.

        Returns:
            List of ContributorInfo objects
        """
        logger.debug("Extracting contributors from git shortlog")

        # Get contributor list with commit counts
        process = await asyncio.create_subprocess_exec(
            "git",
            "shortlog",
            "-sne",
            "--all",
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"git shortlog failed: {error_msg}")

        # Parse shortlog output
        output = stdout.decode("utf-8", errors="replace")
        contributors = self._parse_shortlog(output)

        # Enrich with first/last commit dates and top files
        await self._enrich_contributors(contributors)

        logger.debug("Extracted %d contributors", len(contributors))
        return contributors

    def _parse_shortlog(self, output: str) -> list[ContributorInfo]:
        """Parse git shortlog output.

        Format: "    42  John Doe <john@example.com>"

        Args:
            output: Raw shortlog output

        Returns:
            List of ContributorInfo objects (without enrichment)
        """
        contributors: list[ContributorInfo] = []

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse: "42  John Doe <john@example.com>"
            match = re.match(r"(\d+)\s+(.+?)\s+<(.+?)>", line)
            if match:
                commit_count = int(match.group(1))
                name = match.group(2)
                email = match.group(3)

                contributors.append(
                    ContributorInfo(
                        name=name,
                        email=email,
                        commit_count=commit_count,
                    )
                )

        return contributors

    async def _enrich_contributors(self, contributors: list[ContributorInfo]) -> None:
        """Enrich contributor info with first/last commit dates and top files.

        Modifies contributors in place.

        Args:
            contributors: List of contributors to enrich
        """
        for contributor in contributors:
            # Get first and last commit dates
            process = await asyncio.create_subprocess_exec(
                "git",
                "log",
                "--author",
                contributor.email,
                "--format=%aI",
                "--reverse",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()

            if process.returncode == 0:
                dates = stdout.decode("utf-8", errors="replace").strip().split("\n")
                if dates and dates[0]:
                    try:
                        contributor.first_commit = datetime.fromisoformat(
                            dates[0].replace("Z", "+00:00")
                        )
                        contributor.last_commit = datetime.fromisoformat(
                            dates[-1].replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

            # Get top files
            process = await asyncio.create_subprocess_exec(
                "git",
                "log",
                "--author",
                contributor.email,
                "--name-only",
                "--format=",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()

            if process.returncode == 0:
                files = stdout.decode("utf-8", errors="replace").strip().split("\n")
                file_counts: dict[str, int] = {}

                for file_path in files:
                    if file_path:
                        file_counts[file_path] = file_counts.get(file_path, 0) + 1

                # Get top 5 files
                top_files = sorted(
                    file_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
                contributor.top_files = [f for f, _ in top_files]

    async def _extract_issues(self) -> list[IssueInfo]:
        """Extract GitHub issues via gh CLI.

        Returns:
            List of IssueInfo objects, or empty list if gh CLI unavailable
        """
        logger.debug("Extracting GitHub issues")

        try:
            process = await asyncio.create_subprocess_exec(
                "gh",
                "issue",
                "list",
                "--state",
                "all",
                "--json",
                "number,title,state,labels,createdAt,closedAt,body",
                "--limit",
                "1000",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                logger.warning(
                    "gh issue list failed (gh CLI may not be installed): %s", error_msg
                )
                return []

            # Parse JSON output
            output = stdout.decode("utf-8", errors="replace")
            data = json.loads(output)

            issues: list[IssueInfo] = []
            for issue_data in data:
                # Parse dates
                created_at = None
                closed_at = None

                if issue_data.get("createdAt"):
                    try:
                        created_at = datetime.fromisoformat(
                            issue_data["createdAt"].replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                if issue_data.get("closedAt"):
                    try:
                        closed_at = datetime.fromisoformat(
                            issue_data["closedAt"].replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                # Extract labels
                labels = [
                    label.get("name", "") for label in issue_data.get("labels", [])
                ]

                issues.append(
                    IssueInfo(
                        number=issue_data["number"],
                        title=issue_data["title"],
                        state=issue_data.get("state", "open"),
                        labels=labels,
                        created_at=created_at,
                        closed_at=closed_at,
                        body=issue_data.get("body", ""),
                    )
                )

            logger.debug("Extracted %d issues", len(issues))
            return issues

        except Exception as e:
            logger.warning("Failed to extract GitHub issues: %s", e)
            return []

    async def _extract_pull_requests(self) -> list[PullRequestInfo]:
        """Extract GitHub pull requests via gh CLI.

        Returns:
            List of PullRequestInfo objects, or empty list if gh CLI unavailable
        """
        logger.debug("Extracting GitHub pull requests")

        try:
            process = await asyncio.create_subprocess_exec(
                "gh",
                "pr",
                "list",
                "--state",
                "all",
                "--json",
                "number,title,state,mergedAt,additions,deletions,labels",
                "--limit",
                "1000",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                logger.warning(
                    "gh pr list failed (gh CLI may not be installed): %s", error_msg
                )
                return []

            # Parse JSON output
            output = stdout.decode("utf-8", errors="replace")
            data = json.loads(output)

            prs: list[PullRequestInfo] = []
            for pr_data in data:
                # Parse merged date
                merged_at = None
                if pr_data.get("mergedAt"):
                    try:
                        merged_at = datetime.fromisoformat(
                            pr_data["mergedAt"].replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                # Extract labels
                labels = [label.get("name", "") for label in pr_data.get("labels", [])]

                # Note: gh pr list doesn't return files_changed directly
                # We approximate with additions + deletions as a proxy
                additions = pr_data.get("additions", 0)
                deletions = pr_data.get("deletions", 0)

                prs.append(
                    PullRequestInfo(
                        number=pr_data["number"],
                        title=pr_data["title"],
                        state=pr_data.get("state", "open"),
                        merged_at=merged_at,
                        files_changed=0,  # Not available in this API
                        additions=additions,
                        deletions=deletions,
                        labels=labels,
                    )
                )

            logger.debug("Extracted %d pull requests", len(prs))
            return prs

        except Exception as e:
            logger.warning("Failed to extract GitHub pull requests: %s", e)
            return []

    async def _extract_docs(self) -> list[DocReference]:
        """Extract documentation files from project.

        Scans for:
        - README.md, ARCHITECTURE.md, CONTRIBUTING.md (root)
        - docs/**/*.md
        - Any .md files with "design", "spec", "architecture" in name

        Returns:
            List of DocReference objects
        """
        logger.debug("Extracting documentation files")

        docs: list[DocReference] = []

        # Root-level docs
        root_doc_names = [
            "README.md",
            "ARCHITECTURE.md",
            "CONTRIBUTING.md",
            "DESIGN.md",
            "SPECIFICATION.md",
            "ROADMAP.md",
        ]

        for doc_name in root_doc_names:
            doc_path = self.project_root / doc_name
            if doc_path.exists() and doc_path.is_file():
                docs.append(self._create_doc_reference(doc_path))

        # docs/ directory
        docs_dir = self.project_root / "docs"
        if docs_dir.exists() and docs_dir.is_dir():
            for md_file in docs_dir.rglob("*.md"):
                if md_file.is_file():
                    docs.append(self._create_doc_reference(md_file))

        # Other important docs
        for md_file in self.project_root.rglob("*.md"):
            if md_file.is_file():
                name_lower = md_file.name.lower()
                if any(
                    keyword in name_lower
                    for keyword in ["design", "spec", "architecture", "proposal"]
                ):
                    # Check if not already included
                    if not any(
                        d.path == str(md_file.relative_to(self.project_root))
                        for d in docs
                    ):
                        docs.append(self._create_doc_reference(md_file))

        logger.debug("Extracted %d documentation files", len(docs))
        return docs

    def _create_doc_reference(self, doc_path: Path) -> DocReference:
        """Create DocReference from file path.

        Args:
            doc_path: Absolute path to documentation file

        Returns:
            DocReference object
        """
        # Get relative path
        try:
            rel_path = doc_path.relative_to(self.project_root)
        except ValueError:
            rel_path = doc_path

        # Extract title from first # heading if available
        title = ""
        try:
            content = doc_path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        except Exception:
            pass

        # Fallback to filename
        if not title:
            title = doc_path.stem

        # Count words
        word_count = 0
        try:
            content = doc_path.read_text(encoding="utf-8", errors="replace")
            word_count = len(content.split())
        except Exception:
            pass

        # Get last modified time
        last_modified = None
        try:
            stat = doc_path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            pass

        return DocReference(
            path=str(rel_path),
            title=title,
            word_count=word_count,
            last_modified=last_modified,
        )

    async def get_git_metadata(self) -> dict[str, Any]:
        """Get git metadata (remote, branch, HEAD commit).

        Returns:
            Dict with 'remote', 'branch', 'commit' keys
        """
        metadata: dict[str, Any] = {
            "remote": "",
            "branch": "",
            "commit": "",
        }

        # Get remote URL
        process = await asyncio.create_subprocess_exec(
            "git",
            "config",
            "--get",
            "remote.origin.url",
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()
        if process.returncode == 0:
            metadata["remote"] = stdout.decode("utf-8", errors="replace").strip()

        # Get current branch
        process = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()
        if process.returncode == 0:
            metadata["branch"] = stdout.decode("utf-8", errors="replace").strip()

        # Get HEAD commit
        process = await asyncio.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            cwd=self.project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await process.communicate()
        if process.returncode == 0:
            metadata["commit"] = stdout.decode("utf-8", errors="replace").strip()

        return metadata
