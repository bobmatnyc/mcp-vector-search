"""Unit tests for ReviewEngine."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_vector_search.analysis.review.engine import REVIEW_SEARCH_QUERIES, ReviewEngine
from mcp_vector_search.analysis.review.models import (
    ReviewFinding,
    ReviewResult,
    ReviewType,
    Severity,
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
def review_engine(mock_search_engine, mock_llm_client, tmp_path):
    """Create a ReviewEngine instance with mocked dependencies."""
    return ReviewEngine(
        search_engine=mock_search_engine,
        knowledge_graph=None,
        llm_client=mock_llm_client,
        project_root=tmp_path,
    )


@pytest.mark.asyncio
async def test_run_review_returns_result(review_engine):
    """Test that run_review returns a ReviewResult with correct structure."""
    result = await review_engine.run_review(ReviewType.SECURITY)

    assert isinstance(result, ReviewResult)
    assert result.review_type == ReviewType.SECURITY
    assert isinstance(result.findings, list)
    assert isinstance(result.summary, str)
    assert result.model_used == "test-model"


@pytest.mark.asyncio
async def test_run_review_with_file_filter(review_engine, mock_search_engine):
    """Test that run_review with file_filter calls search engine."""
    # Setup empty search results - just verify filter mechanism works
    mock_search_engine.search.return_value = []

    result = await review_engine.run_review(
        ReviewType.SECURITY, file_filter=["/project/src/auth.py"]
    )

    assert isinstance(result, ReviewResult)
    # Verify search was called
    assert mock_search_engine.search.called


@pytest.mark.asyncio
async def test_run_review_with_use_cache_false(review_engine, mock_search_engine):
    """Test that run_review with use_cache=False sets cache tracking correctly."""
    # Use empty results to avoid LLM call
    mock_search_engine.search.return_value = []

    result = await review_engine.run_review(ReviewType.QUALITY, use_cache=False)

    assert isinstance(result, ReviewResult)
    assert result.cache_hits == 0


def test_review_type_enum_values():
    """Test that ReviewType enum values match expected strings."""
    assert ReviewType.SECURITY.value == "security"
    assert ReviewType.ARCHITECTURE.value == "architecture"
    assert ReviewType.PERFORMANCE.value == "performance"
    assert ReviewType.QUALITY.value == "quality"
    assert ReviewType.TESTING.value == "testing"
    assert ReviewType.DOCUMENTATION.value == "documentation"


def test_review_search_queries_completeness():
    """Test that REVIEW_SEARCH_QUERIES has entries for all ReviewType values."""
    for review_type in ReviewType:
        assert review_type in REVIEW_SEARCH_QUERIES
        assert isinstance(REVIEW_SEARCH_QUERIES[review_type], list)
        assert len(REVIEW_SEARCH_QUERIES[review_type]) > 0


def test_parse_findings_json_with_code_block(review_engine):
    """Test _parse_findings_json handles JSON in markdown code blocks."""
    llm_response = """Here are the findings:

```json
[
  {
    "title": "SQL Injection",
    "description": "User input not sanitized",
    "severity": "critical",
    "file_path": "src/auth.py",
    "start_line": 42,
    "end_line": 45,
    "category": "security",
    "recommendation": "Use parameterized queries",
    "confidence": 0.95
  }
]
```
"""

    findings = review_engine._parse_findings_json(llm_response)

    assert len(findings) == 1
    assert findings[0].title == "SQL Injection"
    assert findings[0].severity == Severity.CRITICAL


def test_parse_findings_json_plain_json(review_engine):
    """Test _parse_findings_json handles plain JSON without code blocks."""
    llm_response = """[
  {
    "title": "Unused Variable",
    "description": "Variable x is defined but never used",
    "severity": "low",
    "file_path": "src/utils.py",
    "start_line": 10,
    "end_line": 10,
    "category": "quality",
    "recommendation": "Remove unused variable",
    "confidence": 0.85
  }
]"""

    findings = review_engine._parse_findings_json(llm_response)

    assert len(findings) == 1
    assert findings[0].title == "Unused Variable"
    assert findings[0].severity == Severity.LOW


def test_parse_findings_json_invalid_json(review_engine):
    """Test _parse_findings_json handles invalid JSON gracefully."""
    llm_response = "This is not JSON at all { invalid }"

    findings = review_engine._parse_findings_json(llm_response)

    # Should return empty list on parse error
    assert len(findings) == 0


def test_parse_findings_json_with_multiline_strings(review_engine):
    """Test _parse_findings_json handles JSON with multiline strings in title/description."""
    llm_response = """Here are the security findings:

```json
[
  {
    "title": "Authentication Issue with
    Session Management",
    "description": "The authentication system has a vulnerability where
    sessions are not properly validated, allowing unauthorized access
    to protected resources.",
    "severity": "high",
    "file_path": "src/auth/session.py",
    "start_line": 42,
    "end_line": 58,
    "category": "security",
    "recommendation": "Implement proper session validation
    and add CSRF protection",
    "confidence": 0.9
  }
]
```"""

    findings = review_engine._parse_findings_json(llm_response)

    assert len(findings) == 1
    assert "Authentication Issue with" in findings[0].title
    assert "Session Management" in findings[0].title
    assert "authentication system has a vulnerability" in findings[0].description
    assert findings[0].severity == Severity.HIGH
    assert findings[0].file_path == "src/auth/session.py"


def test_parse_findings_json_with_embedded_code_blocks(review_engine):
    """Test _parse_findings_json handles JSON containing code blocks in descriptions."""
    llm_response = """```json
[
  {
    "title": "SQL Injection Vulnerability",
    "description": "The query is vulnerable:\\n```python\\nquery = 'SELECT * FROM users WHERE name = ' + user_input\\n```\\nThis allows SQL injection attacks.",
    "severity": "critical",
    "file_path": "src/db/queries.py",
    "start_line": 15,
    "end_line": 15,
    "category": "security",
    "recommendation": "Use parameterized queries",
    "confidence": 0.95
  }
]
```"""

    findings = review_engine._parse_findings_json(llm_response)

    assert len(findings) == 1
    assert findings[0].title == "SQL Injection Vulnerability"
    assert "SELECT * FROM users" in findings[0].description
    assert "SQL injection attacks" in findings[0].description
    assert findings[0].severity == Severity.CRITICAL


def test_format_code_context_returns_non_empty(review_engine):
    """Test _format_code_context returns non-empty string with code chunks."""
    mock_result = MagicMock()
    mock_result.file_path = "src/test.py"
    mock_result.start_line = 1
    mock_result.end_line = 10
    mock_result.function_name = "test_func"
    mock_result.similarity_score = 0.9
    mock_result.language = "python"
    mock_result.content = "def test_func():\n    pass"

    context = review_engine._format_code_context([mock_result])

    assert isinstance(context, str)
    assert len(context) > 0
    assert "src/test.py" in context
    assert "test_func" in context
    assert "def test_func():" in context


def test_format_code_context_empty_results(review_engine):
    """Test _format_code_context with empty results."""
    context = review_engine._format_code_context([])

    assert isinstance(context, str)
    # Empty list should return empty string
    assert context == ""


def test_format_kg_context_with_relationships(review_engine):
    """Test _format_kg_context with knowledge graph relationships."""
    relationships = [
        {
            "entity": "login",
            "type": "function",
            "file": "src/auth.py",
            "related": [
                {"relationship": "calls", "name": "validate_password"},
                {"relationship": "used_by", "name": "authenticate"},
            ],
        }
    ]

    context = review_engine._format_kg_context(relationships)

    assert isinstance(context, str)
    assert "login" in context
    assert "validate_password" in context


def test_format_kg_context_empty_relationships(review_engine):
    """Test _format_kg_context with no relationships."""
    context = review_engine._format_kg_context([])

    assert isinstance(context, str)
    assert "No knowledge graph relationships available" in context


def test_generate_summary_no_findings(review_engine):
    """Test _generate_summary with no findings."""
    summary = review_engine._generate_summary([], ReviewType.SECURITY)

    assert "No security issues found" in summary


def test_generate_summary_with_findings(review_engine):
    """Test _generate_summary with findings."""
    findings = [
        ReviewFinding(
            title="Issue 1",
            description="Test",
            severity=Severity.CRITICAL,
            file_path="test.py",
            start_line=1,
            end_line=1,
            category="security",
            recommendation="Fix it",
            confidence=0.9,
        ),
        ReviewFinding(
            title="Issue 2",
            description="Test",
            severity=Severity.LOW,
            file_path="test.py",
            start_line=2,
            end_line=2,
            category="style",
            recommendation="Improve",
            confidence=0.7,
        ),
    ]

    summary = review_engine._generate_summary(findings, ReviewType.SECURITY)

    assert "Found 2 security issue(s)" in summary
    assert "CRITICAL: 1" in summary
    assert "LOW: 1" in summary


@pytest.mark.asyncio
async def test_run_review_no_chunks_found(review_engine, mock_search_engine):
    """Test run_review when no code chunks are found."""
    mock_search_engine.search.return_value = []

    result = await review_engine.run_review(ReviewType.PERFORMANCE)

    assert result.findings == []
    assert "No code chunks found" in result.summary
    assert result.context_chunks_used == 0


def test_get_chunk_id_with_chunk_id_attr(review_engine):
    """Test _get_chunk_id when result has chunk_id attribute."""
    mock_result = MagicMock()
    mock_result.chunk_id = "test-chunk-123"
    mock_result.file_path = "test.py"
    mock_result.start_line = 1
    mock_result.end_line = 10

    chunk_id = review_engine._get_chunk_id(mock_result)

    assert chunk_id == "test-chunk-123"


def test_get_chunk_id_fallback(review_engine):
    """Test _get_chunk_id fallback to file path and line range."""
    mock_result = MagicMock()
    mock_result.chunk_id = None
    mock_result.file_path = "src/test.py"
    mock_result.start_line = 5
    mock_result.end_line = 15

    chunk_id = review_engine._get_chunk_id(mock_result)

    assert chunk_id == "src/test.py:5-15"


@pytest.mark.asyncio
async def test_gather_kg_relationships_without_kg(review_engine):
    """Test _gather_kg_relationships when knowledge_graph is None."""
    mock_result = MagicMock()
    mock_result.function_name = "test_func"
    mock_result.file_path = "test.py"

    relationships = await review_engine._gather_kg_relationships([mock_result])

    # Should return empty list when KG is None
    assert relationships == []


def test_parse_findings_json_with_optional_fields(review_engine):
    """Test _parse_findings_json handles findings with optional fields."""
    llm_response = json.dumps(
        [
            {
                "title": "Security Issue",
                "description": "Test",
                "severity": "high",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 1,
                "category": "security",
                "recommendation": "Fix",
                "confidence": 0.9,
                "cwe_id": "CWE-89",
                "code_snippet": "SELECT * FROM users",
                "related_files": ["auth.py", "db.py"],
            }
        ]
    )

    findings = review_engine._parse_findings_json(llm_response)

    assert len(findings) == 1
    assert findings[0].cwe_id == "CWE-89"
    assert findings[0].code_snippet == "SELECT * FROM users"
    assert findings[0].related_files == ["auth.py", "db.py"]


def test_parse_findings_json_with_missing_required_field(review_engine):
    """Test _parse_findings_json skips findings with missing required fields."""
    llm_response = json.dumps(
        [
            {
                # Missing "title" field
                "description": "Test",
                "severity": "low",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 1,
                "category": "quality",
                "recommendation": "Fix",
                "confidence": 0.8,
            }
        ]
    )

    findings = review_engine._parse_findings_json(llm_response)

    # Should skip invalid finding
    assert len(findings) == 0


def test_cache_findings(review_engine):
    """Test _cache_findings caches findings per chunk."""
    findings = [
        ReviewFinding(
            title="Issue",
            description="Test",
            severity=Severity.MEDIUM,
            file_path="/tmp/test.py",
            start_line=5,
            end_line=8,
            category="quality",
            recommendation="Fix",
            confidence=0.8,
        )
    ]

    mock_result = MagicMock()
    mock_result.file_path = "/tmp/test.py"
    mock_result.content = "test content"
    mock_result.start_line = 1
    mock_result.end_line = 10

    # Should not raise error
    review_engine._cache_findings(findings, [mock_result], ReviewType.QUALITY)


@pytest.mark.asyncio
async def test_run_review_with_scope(review_engine, mock_search_engine):
    """Test run_review with scope parameter."""
    mock_search_engine.search.return_value = []

    result = await review_engine.run_review(ReviewType.SECURITY, scope="src/auth")

    assert isinstance(result, ReviewResult)
    assert result.scope == "src/auth"


def test_severity_enum_values():
    """Test Severity enum values."""
    assert Severity.CRITICAL.value == "critical"
    assert Severity.HIGH.value == "high"
    assert Severity.MEDIUM.value == "medium"
    assert Severity.LOW.value == "low"
    assert Severity.INFO.value == "info"
