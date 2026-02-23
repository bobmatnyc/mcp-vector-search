"""Unit tests for PRReviewEngine."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_vector_search.analysis.review.models import Severity
from mcp_vector_search.analysis.review.pr_engine import PRReviewEngine
from mcp_vector_search.analysis.review.pr_models import (
    PRContext,
    PRFilePatch,
    PRReviewComment,
    PRReviewResult,
)


@pytest.fixture
def mock_search_engine():
    """Create a mock search engine."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.model = "test-model"
    mock._chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "[]"}}]}
    )
    return mock


@pytest.fixture
def pr_engine(mock_search_engine, mock_llm_client, tmp_path):
    """Create a PRReviewEngine instance with mocked dependencies."""
    return PRReviewEngine(
        search_engine=mock_search_engine,
        knowledge_graph=None,
        llm_client=mock_llm_client,
        project_root=tmp_path,
    )


def test_pr_engine_initializes_correctly(pr_engine):
    """Test PRReviewEngine initializes with all required components."""
    assert pr_engine.search_engine is not None
    assert pr_engine.knowledge_graph is None
    assert pr_engine.llm_client is not None
    assert pr_engine.project_root is not None
    assert pr_engine.instructions_loader is not None


@pytest.mark.asyncio
async def test_build_language_context_python(pr_engine):
    """Test _build_language_context returns language-specific content for Python."""
    patches = [
        PRFilePatch(
            file_path="src/auth.py",
            old_content="def login(): pass",
            new_content="def login(user): pass",
            diff_text="@@ -1,1 +1,1 @@\n-def login(): pass\n+def login(user): pass",
            additions=1,
            deletions=1,
            is_new_file=False,
            is_deleted=False,
            is_renamed=False,
            old_path=None,
        )
    ]

    context = await pr_engine._build_language_context(patches)

    assert isinstance(context, str)
    assert "Python" in context
    # Should contain language-specific standards
    assert len(context) > 0


@pytest.mark.asyncio
async def test_build_language_context_multiple_languages(pr_engine):
    """Test _build_language_context handles multiple languages."""
    patches = [
        PRFilePatch(
            file_path="src/main.py",
            old_content=None,
            new_content="def main(): pass",
            diff_text="@@ -0,0 +1,1 @@\n+def main(): pass",
            additions=1,
            deletions=0,
            is_new_file=True,
            is_deleted=False,
            is_renamed=False,
            old_path=None,
        ),
        PRFilePatch(
            file_path="src/utils.ts",
            old_content=None,
            new_content="function util() {}",
            diff_text="@@ -0,0 +1,1 @@\n+function util() {}",
            additions=1,
            deletions=0,
            is_new_file=True,
            is_deleted=False,
            is_renamed=False,
            old_path=None,
        ),
    ]

    context = await pr_engine._build_language_context(patches)

    assert isinstance(context, str)
    # Should mention both languages
    assert "Python" in context
    assert "TypeScript" in context


@pytest.mark.asyncio
async def test_build_language_context_no_known_languages(pr_engine):
    """Test _build_language_context with unknown file extensions."""
    patches = [
        PRFilePatch(
            file_path="README.md",
            old_content="# Old",
            new_content="# New",
            diff_text="@@ -1,1 +1,1 @@\n-# Old\n+# New",
            additions=1,
            deletions=1,
            is_new_file=False,
            is_deleted=False,
            is_renamed=False,
            old_path=None,
        )
    ]

    context = await pr_engine._build_language_context(patches)

    # Should return empty string for unknown languages
    assert context == ""


def test_parse_diff_creates_pr_context(pr_engine):
    """Test _parse_git_diff correctly splits a unified diff into PRFilePatch objects."""
    diff_text = """diff --git a/src/auth.py b/src/auth.py
index 123abc..456def 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,5 +1,6 @@
 def login(username, password):
+    # Added validation
     if not username:
         return None"""

    mock_git_manager = MagicMock()
    mock_git_manager.parse_diff_stats.return_value = {"src/auth.py": (1, 0)}
    mock_git_manager.get_file_at_ref.return_value = (
        "def login(username, password):\n    if not username:\n        return None"
    )
    mock_git_manager.get_current_branch.return_value = "feature"

    context = pr_engine._parse_git_diff(diff_text, "main", "HEAD", mock_git_manager)

    assert isinstance(context, PRContext)
    assert len(context.patches) > 0
    assert context.base_branch == "main"


@pytest.mark.asyncio
async def test_review_pr_with_empty_patches(pr_engine):
    """Test review_pr with empty patches returns valid PRReviewResult."""
    context = PRContext(
        title="Test PR",
        description="Test description",
        base_branch="main",
        head_branch="feature",
        patches=[],
    )

    result = await pr_engine.review_pr(context)

    assert isinstance(result, PRReviewResult)
    assert result.verdict in ["approve", "request_changes", "comment"]
    assert result.blocking_issues == 0


@pytest.mark.asyncio
async def test_review_pr_returns_pr_review_result(pr_engine, mock_llm_client):
    """Test review_pr returns complete PRReviewResult."""
    patches = [
        PRFilePatch(
            file_path="src/test.py",
            old_content="def test(): pass",
            new_content="def test(x): pass",
            diff_text="@@ -1,1 +1,1 @@\n-def test(): pass\n+def test(x): pass",
            additions=1,
            deletions=1,
            is_new_file=False,
            is_deleted=False,
            is_renamed=False,
            old_path=None,
        )
    ]

    context = PRContext(
        title="Add parameter",
        description="Adds parameter x to test function",
        base_branch="main",
        head_branch="feature",
        patches=patches,
    )

    result = await pr_engine.review_pr(context)

    assert isinstance(result, PRReviewResult)
    assert isinstance(result.summary, str)
    assert isinstance(result.comments, list)
    assert result.model_used == "test-model"
    assert result.duration_seconds >= 0


def test_generate_verdict_no_comments(pr_engine):
    """Test _generate_verdict with no comments returns approve."""
    verdict, score = pr_engine._generate_verdict([])

    assert verdict == "approve"
    assert score == 1.0


def test_generate_verdict_with_blocking_issues(pr_engine):
    """Test _generate_verdict with blocking issues returns request_changes."""
    comments = [
        PRReviewComment(
            file_path="test.py",
            line_number=1,
            comment="Critical issue",
            severity=Severity.CRITICAL,
            category="security",
            suggestion=None,
            is_blocking=True,
        )
    ]

    verdict, score = pr_engine._generate_verdict(comments)

    assert verdict == "request_changes"
    assert score < 1.0


def test_generate_verdict_with_non_blocking_warnings(pr_engine):
    """Test _generate_verdict with non-blocking warnings returns comment."""
    comments = [
        PRReviewComment(
            file_path="test.py",
            line_number=1,
            comment="Medium issue",
            severity=Severity.MEDIUM,
            category="quality",
            suggestion=None,
            is_blocking=False,
        )
    ]

    verdict, score = pr_engine._generate_verdict(comments)

    assert verdict == "comment"
    assert 0.0 < score < 1.0


def test_generate_verdict_with_low_severity_only(pr_engine):
    """Test _generate_verdict with only low severity returns approve."""
    comments = [
        PRReviewComment(
            file_path="test.py",
            line_number=1,
            comment="Style suggestion",
            severity=Severity.LOW,
            category="style",
            suggestion=None,
            is_blocking=False,
        )
    ]

    verdict, score = pr_engine._generate_verdict(comments)

    assert verdict == "approve"
    assert score > 0.9


def test_generate_summary_no_comments(pr_engine):
    """Test _generate_summary with no comments."""
    summary = pr_engine._generate_summary([])

    assert "No issues found" in summary or "looks good" in summary.lower()


def test_generate_summary_with_comments(pr_engine):
    """Test _generate_summary with various severity levels."""
    comments = [
        PRReviewComment(
            file_path="test.py",
            line_number=1,
            comment="Critical",
            severity=Severity.CRITICAL,
            category="security",
            suggestion=None,
            is_blocking=True,
        ),
        PRReviewComment(
            file_path="test.py",
            line_number=2,
            comment="Info",
            severity=Severity.INFO,
            category="style",
            suggestion=None,
            is_blocking=False,
        ),
    ]

    summary = pr_engine._generate_summary(comments)

    assert "2 issue(s)" in summary
    assert "CRITICAL" in summary
    assert "INFO" in summary


def test_parse_comments_json_with_code_block(pr_engine):
    """Test _parse_comments_json handles JSON in code blocks."""
    llm_response = """```json
[
  {
    "file_path": "test.py",
    "line_number": 42,
    "comment": "SQL injection risk",
    "severity": "critical",
    "category": "security",
    "suggestion": "Use parameterized queries",
    "is_blocking": true
  }
]
```"""

    comments = pr_engine._parse_comments_json(llm_response)

    assert len(comments) == 1
    assert comments[0].file_path == "test.py"
    assert comments[0].severity == Severity.CRITICAL
    assert comments[0].is_blocking is True


def test_parse_comments_json_plain_json(pr_engine):
    """Test _parse_comments_json handles plain JSON."""
    llm_response = json.dumps(
        [
            {
                "file_path": "test.py",
                "line_number": 10,
                "comment": "Consider refactoring",
                "severity": "low",
                "category": "quality",
                "suggestion": None,
                "is_blocking": False,
            }
        ]
    )

    comments = pr_engine._parse_comments_json(llm_response)

    assert len(comments) == 1
    assert comments[0].severity == Severity.LOW


def test_parse_comments_json_invalid_json(pr_engine):
    """Test _parse_comments_json handles invalid JSON gracefully."""
    llm_response = "This is not valid JSON {{"

    comments = pr_engine._parse_comments_json(llm_response)

    # Should return empty list
    assert comments == []


def test_parse_comments_json_with_overall_comment(pr_engine):
    """Test _parse_comments_json handles overall PR comments (null file_path)."""
    llm_response = json.dumps(
        [
            {
                "file_path": None,
                "line_number": None,
                "comment": "Overall looks good",
                "severity": "info",
                "category": "quality",
                "suggestion": None,
                "is_blocking": False,
            }
        ]
    )

    comments = pr_engine._parse_comments_json(llm_response)

    assert len(comments) == 1
    assert comments[0].file_path is None
    assert comments[0].line_number is None


def test_extract_function_names_python(pr_engine):
    """Test _extract_function_names extracts Python function names."""
    diff_text = """@@ -1,5 +1,7 @@
+def new_function(x):
+    return x * 2
 def existing_function():
     pass"""

    functions = pr_engine._extract_function_names(diff_text)

    assert "new_function" in functions


def test_extract_function_names_javascript(pr_engine):
    """Test _extract_function_names extracts JavaScript function names."""
    diff_text = """@@ -1,3 +1,5 @@
+function handleClick() {
+    console.log("clicked");
+}"""

    functions = pr_engine._extract_function_names(diff_text)

    assert "handleClick" in functions


def test_extract_function_names_empty_diff(pr_engine):
    """Test _extract_function_names with empty diff."""
    functions = pr_engine._extract_function_names("")

    assert functions == []


@pytest.mark.asyncio
async def test_gather_file_contexts_skips_deleted_files(pr_engine):
    """Test _gather_file_contexts skips deleted files."""
    patches = [
        PRFilePatch(
            file_path="deleted.py",
            old_content="def old(): pass",
            new_content=None,
            diff_text="",
            additions=0,
            deletions=1,
            is_new_file=False,
            is_deleted=True,
            is_renamed=False,
            old_path=None,
        )
    ]

    contexts = await pr_engine._gather_file_contexts(patches)

    # Deleted files should be skipped
    assert len(contexts) == 0


@pytest.mark.asyncio
async def test_gather_file_contexts_includes_related_files(
    pr_engine, mock_search_engine
):
    """Test _gather_file_contexts includes related files from search."""
    mock_result = MagicMock()
    mock_result.file_path = "related.py"
    mock_result.function_name = "related_func"
    mock_result.similarity_score = 0.8

    mock_search_engine.search.return_value = [mock_result]

    patches = [
        PRFilePatch(
            file_path="src/test.py",
            old_content=None,
            new_content="def test(): pass",
            diff_text="",
            additions=1,
            deletions=0,
            is_new_file=True,
            is_deleted=False,
            is_renamed=False,
            old_path=None,
        )
    ]

    contexts = await pr_engine._gather_file_contexts(patches)

    assert len(contexts) == 1
    assert len(contexts[0]["related_files"]) > 0


def test_format_file_contexts(pr_engine):
    """Test _format_file_contexts produces formatted text."""
    contexts = [
        {
            "file_path": "test.py",
            "diff": "diff content",
            "additions": 5,
            "deletions": 2,
            "related_files": [],
            "kg_relationships": [],
        }
    ]

    formatted = pr_engine._format_file_contexts(contexts)

    assert isinstance(formatted, str)
    assert "test.py" in formatted
    assert "+5" in formatted or "5" in formatted


def test_format_file_contexts_truncates_large_diffs(pr_engine):
    """Test _format_file_contexts truncates very large diffs."""
    large_diff = "\n".join([f"+line {i}" for i in range(200)])

    contexts = [
        {
            "file_path": "large.py",
            "diff": large_diff,
            "additions": 200,
            "deletions": 0,
            "related_files": [],
            "kg_relationships": [],
        }
    ]

    formatted = pr_engine._format_file_contexts(contexts)

    assert "truncated" in formatted.lower()


@pytest.mark.asyncio
async def test_review_pr_with_custom_instructions(pr_engine):
    """Test review_pr uses custom instructions when provided."""
    context = PRContext(
        title="Test PR",
        description=None,
        base_branch="main",
        head_branch="feature",
        patches=[],
    )

    result = await pr_engine.review_pr(
        context, custom_instructions="Custom instruction: Check for X"
    )

    assert result.review_instructions_applied == "custom (override)"


@pytest.mark.asyncio
async def test_review_pr_counts_blocking_issues(pr_engine, mock_llm_client):
    """Test review_pr correctly counts blocking issues."""
    # Mock LLM to return blocking issue
    mock_llm_client._chat_completion = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {
                                    "file_path": "test.py",
                                    "line_number": 1,
                                    "comment": "Critical issue",
                                    "severity": "critical",
                                    "category": "security",
                                    "suggestion": None,
                                    "is_blocking": True,
                                }
                            ]
                        )
                    }
                }
            ]
        }
    )

    context = PRContext(
        title="Test PR",
        description=None,
        base_branch="main",
        head_branch="feature",
        patches=[],
    )

    result = await pr_engine.review_pr(context)

    assert result.blocking_issues == 1


def test_pr_file_patch_immutable():
    """Test PRFilePatch is immutable (frozen dataclass)."""
    patch = PRFilePatch(
        file_path="test.py",
        old_content="old",
        new_content="new",
        diff_text="diff",
        additions=1,
        deletions=1,
        is_new_file=False,
        is_deleted=False,
        is_renamed=False,
        old_path=None,
    )

    # Should not be able to modify
    with pytest.raises(Exception):  # FrozenInstanceError
        patch.file_path = "modified.py"


def test_pr_context_immutable():
    """Test PRContext is immutable (frozen dataclass)."""
    context = PRContext(
        title="Test",
        description=None,
        base_branch="main",
        head_branch="feature",
        patches=[],
    )

    # Should not be able to modify
    with pytest.raises(Exception):  # FrozenInstanceError
        context.title = "Modified"


def test_pr_review_comment_immutable():
    """Test PRReviewComment is immutable (frozen dataclass)."""
    comment = PRReviewComment(
        file_path="test.py",
        line_number=1,
        comment="Test",
        severity=Severity.LOW,
        category="style",
        suggestion=None,
        is_blocking=False,
    )

    # Should not be able to modify
    with pytest.raises(Exception):  # FrozenInstanceError
        comment.comment = "Modified"


def test_pr_review_result_immutable():
    """Test PRReviewResult is immutable (frozen dataclass)."""
    result = PRReviewResult(
        summary="Test",
        verdict="approve",
        overall_score=1.0,
        comments=[],
        blocking_issues=0,
        warnings=0,
        suggestions=0,
        context_files_used=0,
        kg_relationships_used=0,
        review_instructions_applied=None,
        model_used="test",
        duration_seconds=0.0,
    )

    # Should not be able to modify
    with pytest.raises(Exception):  # FrozenInstanceError
        result.verdict = "reject"
