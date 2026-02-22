"""Data models for pull request / merge request code review.

This module defines the data structures used for PR/MR review functionality,
including file patches, review comments, and complete review results.

Design Philosophy:
    - Immutable dataclasses for thread safety
    - Type hints for all fields
    - Support for inline and overall comments
    - Severity and blocking semantics for CI/CD integration
    - Metadata tracking for reproducibility
"""

from dataclasses import dataclass, field

from .models import Severity


@dataclass(frozen=True)
class PRFilePatch:
    """A single file's changes in a pull request.

    Represents the diff for one file, including metadata about the change type
    (new file, deleted, renamed) and the actual diff content.

    Attributes:
        file_path: Path to the file relative to project root
        old_content: Full file content before changes (None for new files)
        new_content: Full file content after changes (None for deleted files)
        diff_text: Unified diff format text (output from git diff)
        additions: Number of lines added
        deletions: Number of lines deleted
        is_new_file: True if this is a newly created file
        is_deleted: True if this file was deleted
        is_renamed: True if this file was renamed/moved
        old_path: Original file path before rename (None if not renamed)

    Example:
        >>> patch = PRFilePatch(
        ...     file_path="src/auth.py",
        ...     old_content="def login(): pass",
        ...     new_content="def login(user): pass",
        ...     diff_text="@@ -1,1 +1,1 @@...",
        ...     additions=1,
        ...     deletions=1,
        ...     is_new_file=False,
        ...     is_deleted=False,
        ...     is_renamed=False,
        ...     old_path=None
        ... )
    """

    file_path: str
    old_content: str | None
    new_content: str | None
    diff_text: str
    additions: int
    deletions: int
    is_new_file: bool
    is_deleted: bool
    is_renamed: bool
    old_path: str | None


@dataclass(frozen=True)
class PRContext:
    """Complete context for a pull request review.

    Aggregates all information needed to perform a comprehensive PR review,
    including metadata, all file changes, and optional labels/author info.

    Attributes:
        title: PR title (one-line summary)
        description: PR description/body (can be None)
        base_branch: Target branch (e.g., "main", "develop")
        head_branch: Source branch being merged
        patches: List of all file changes in the PR
        author: PR author username/email (optional)
        labels: List of PR labels (e.g., ["bugfix", "security"])

    Example:
        >>> context = PRContext(
        ...     title="Fix authentication bug",
        ...     description="Resolves #123 by adding input validation",
        ...     base_branch="main",
        ...     head_branch="fix-auth-bug",
        ...     patches=[patch1, patch2],
        ...     author="john@example.com",
        ...     labels=["bugfix", "security"]
        ... )
    """

    title: str
    description: str | None
    base_branch: str
    head_branch: str
    patches: list[PRFilePatch]
    author: str | None = None
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PRReviewComment:
    """A single review comment on a pull request.

    Represents either an inline comment (attached to specific file/line) or
    an overall PR-level comment. Includes severity and blocking semantics
    for automated merge policies.

    Attributes:
        file_path: Path to file (None for overall PR comments)
        line_number: Line number in file (None for overall/file-level comments)
        comment: The review comment text
        severity: Severity level (info, low, medium, high, critical)
        category: Issue category (e.g., "security", "style", "logic", "performance")
        suggestion: Optional code fix suggestion (unified diff format or plain code)
        is_blocking: Whether this issue should block PR merge

    Severity to Blocking Mapping:
        - critical: Always blocking (security vulnerabilities, data loss risks)
        - high: Usually blocking (correctness issues, major bugs)
        - medium: Sometimes blocking (depending on team policy)
        - low: Rarely blocking (code style, minor improvements)
        - info: Never blocking (informational comments)

    Example:
        >>> comment = PRReviewComment(
        ...     file_path="src/auth.py",
        ...     line_number=42,
        ...     comment="SQL injection vulnerability detected",
        ...     severity=Severity.CRITICAL,
        ...     category="security",
        ...     suggestion="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        ...     is_blocking=True
        ... )
    """

    file_path: str | None
    line_number: int | None
    comment: str
    severity: Severity
    category: str
    suggestion: str | None
    is_blocking: bool


@dataclass(frozen=True)
class PRReviewResult:
    """Complete result of a pull request review.

    Aggregates all review findings, metadata about the review process,
    and a final verdict recommendation.

    Attributes:
        summary: High-level summary of review findings
        verdict: Review verdict ("approve", "request_changes", "comment")
        overall_score: Quality score (0.0 = worst, 1.0 = best)
        comments: List of all review comments (inline + overall)
        blocking_issues: Count of blocking issues (prevents merge)
        warnings: Count of warnings (severity medium/high, non-blocking)
        suggestions: Count of suggestions (severity low/info)
        context_files_used: Number of codebase files used as context
        kg_relationships_used: Number of knowledge graph relationships queried
        review_instructions_applied: Name/description of custom instructions used
        model_used: LLM model used for review
        duration_seconds: Total review duration

    Verdict Semantics:
        - "approve": No blocking issues, ready to merge
        - "request_changes": Has blocking issues, requires fixes
        - "comment": Has suggestions but no blockers, optional fixes

    Example:
        >>> result = PRReviewResult(
        ...     summary="Found 2 security issues and 3 style improvements",
        ...     verdict="request_changes",
        ...     overall_score=0.65,
        ...     comments=[comment1, comment2],
        ...     blocking_issues=2,
        ...     warnings=0,
        ...     suggestions=3,
        ...     context_files_used=15,
        ...     kg_relationships_used=8,
        ...     review_instructions_applied="company-python-standards",
        ...     model_used="claude-3-5-sonnet-20241022",
        ...     duration_seconds=12.4
        ... )
    """

    summary: str
    verdict: str  # "approve", "request_changes", "comment"
    overall_score: float  # 0.0-1.0
    comments: list[PRReviewComment]
    blocking_issues: int
    warnings: int
    suggestions: int
    context_files_used: int
    kg_relationships_used: int
    review_instructions_applied: str | None
    model_used: str
    duration_seconds: float
