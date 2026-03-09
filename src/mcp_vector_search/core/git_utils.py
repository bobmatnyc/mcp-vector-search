"""Git utilities for temporal knowledge graph queries.

These helpers wrap git subprocess calls used by temporal KG methods.
`shutil.which("git")` is used throughout for bandit B607 compliance
(no bare string "git" as the executable).
"""

import shutil
import subprocess
from pathlib import Path

_GIT = shutil.which("git") or "git"


def is_ancestor_commit(earlier_sha: str, later_sha: str, repo_root: Path) -> bool:
    """Return True if earlier_sha is an ancestor of (or equal to) later_sha.

    Uses `git merge-base --is-ancestor <A> <B>` which exits 0 when A is an
    ancestor of B (or A == B), and exits 1 otherwise.

    Args:
        earlier_sha: The commit that should be the ancestor.
        later_sha: The commit that should be the descendant.
        repo_root: Repository root directory to run git in.

    Returns:
        True if earlier_sha is an ancestor of later_sha (or they are equal).
        False if not, or if either SHA is empty/invalid.
    """
    if not earlier_sha or not later_sha:
        return False

    result = subprocess.run(  # noqa: S603
        [_GIT, "merge-base", "--is-ancestor", earlier_sha, later_sha],
        cwd=repo_root,
        capture_output=True,
    )
    return result.returncode == 0


def get_commit_timestamp(sha: str, repo_root: Path) -> int | None:
    """Return Unix timestamp of a commit, or None if not found.

    Args:
        sha: Git commit SHA.
        repo_root: Repository root directory.

    Returns:
        Unix timestamp as int, or None if the SHA is unknown or git fails.
    """
    if not sha:
        return None

    result = subprocess.run(  # noqa: S603
        [_GIT, "log", "-1", "--format=%ct", sha],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return int(result.stdout.strip())
    return None
