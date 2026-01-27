"""Unit tests for dead code formatters."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from mcp_vector_search.analysis.dead_code import (
    Confidence,
    DeadCodeFinding,
    DeadCodeReport,
)
from mcp_vector_search.analysis.dead_code_formatters import (
    ConsoleFormatter,
    JsonFormatter,
    MarkdownFormatter,
    SarifFormatter,
    get_formatter,
)
from mcp_vector_search.analysis.entry_points import EntryPoint, EntryPointType


@pytest.fixture
def sample_report() -> DeadCodeReport:
    """Create a sample dead code report for testing."""
    entry_points = [
        EntryPoint(
            name="main",
            file_path="src/main.py",
            line_number=10,
            type=EntryPointType.MAIN,
            confidence=1.0,
        ),
        EntryPoint(
            name="test_integration",
            file_path="tests/test_api.py",
            line_number=20,
            type=EntryPointType.TEST,
            confidence=1.0,
        ),
    ]

    findings = [
        DeadCodeFinding(
            function_name="_legacy_converter",
            file_path="utils/converters.py",
            start_line=45,
            end_line=67,
            confidence=Confidence.HIGH,
            reason="Private function with no incoming calls",
            caveats=[],
        ),
        DeadCodeFinding(
            function_name="public_utility",
            file_path="utils/misc.py",
            start_line=89,
            end_line=102,
            confidence=Confidence.MEDIUM,
            reason="No incoming calls detected",
            caveats=["Public function - might be part of API"],
        ),
        DeadCodeFinding(
            function_name="callback_handler",
            file_path="handlers/callbacks.py",
            start_line=30,
            end_line=45,
            confidence=Confidence.LOW,
            reason="Function with decorators is not called",
            caveats=["Has decorators ['@register'] - might be dynamically registered"],
        ),
    ]

    return DeadCodeReport(
        entry_points=entry_points,
        findings=findings,
        total_functions=100,
        reachable_count=97,
        unreachable_count=3,
    )


def test_console_formatter(sample_report: DeadCodeReport) -> None:
    """Test console formatter output."""
    formatter = ConsoleFormatter()
    output = StringIO()

    formatter.format(sample_report, output)
    result = output.getvalue()

    # Check header
    assert "Dead Code Analysis Report" in result
    assert "═" in result

    # Check entry points section (may have ANSI codes)
    assert "Entry Points Detected:" in result
    assert "2" in result
    assert "main" in result
    assert "src/main.py" in result
    assert "10" in result

    # Check findings section
    assert "Dead Code Findings:" in result
    assert "3" in result
    assert "_legacy_converter" in result
    assert "public_utility" in result
    assert "callback_handler" in result

    # Check confidence levels
    assert "HIGH CONFIDENCE" in result
    assert "MEDIUM CONFIDENCE" in result
    assert "LOW CONFIDENCE" in result

    # Check summary section
    assert "Summary" in result
    assert "Total Functions" in result
    assert "100" in result
    assert "97.0%" in result


def test_json_formatter(sample_report: DeadCodeReport) -> None:
    """Test JSON formatter output."""
    formatter = JsonFormatter()
    output = StringIO()

    formatter.format(sample_report, output)
    result = output.getvalue()

    # Parse JSON
    data = json.loads(result)

    # Check structure
    assert "entry_points" in data
    assert "findings" in data
    assert "summary" in data

    # Check entry points
    assert len(data["entry_points"]) == 2
    assert data["entry_points"][0]["name"] == "main"
    assert data["entry_points"][0]["type"] == "MAIN"

    # Check findings
    assert len(data["findings"]) == 3
    assert data["findings"][0]["function_name"] == "_legacy_converter"
    assert data["findings"][0]["confidence"] == "HIGH"
    assert data["findings"][1]["confidence"] == "MEDIUM"
    assert data["findings"][2]["confidence"] == "LOW"

    # Check summary
    assert data["summary"]["total_functions"] == 100
    assert data["summary"]["reachable"] == 97
    assert data["summary"]["unreachable"] == 3
    assert data["summary"]["reachable_percentage"] == 97.0


def test_sarif_formatter(sample_report: DeadCodeReport) -> None:
    """Test SARIF formatter output."""
    formatter = SarifFormatter(tool_version="1.1.28")
    output = StringIO()

    formatter.format(sample_report, output)
    result = output.getvalue()

    # Parse SARIF JSON
    data = json.loads(result)

    # Check SARIF structure
    assert data["$schema"]
    assert data["version"] == "2.1.0"
    assert "runs" in data
    assert len(data["runs"]) == 1

    run = data["runs"][0]

    # Check tool info
    assert run["tool"]["driver"]["name"] == "mcp-vector-search"
    assert run["tool"]["driver"]["version"] == "1.1.28"

    # Check rules
    rules = run["tool"]["driver"]["rules"]
    assert len(rules) == 3
    rule_ids = [r["id"] for r in rules]
    assert "dead-code-high" in rule_ids
    assert "dead-code-medium" in rule_ids
    assert "dead-code-low" in rule_ids

    # Check results
    results = run["results"]
    assert len(results) == 3

    # Check first result (HIGH confidence)
    result_high = results[0]
    assert result_high["ruleId"] == "dead-code-high"
    assert "_legacy_converter" in result_high["message"]["text"]
    assert (
        result_high["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        == "utils/converters.py"
    )
    assert result_high["locations"][0]["physicalLocation"]["region"]["startLine"] == 45
    assert result_high["locations"][0]["physicalLocation"]["region"]["endLine"] == 67


def test_markdown_formatter(sample_report: DeadCodeReport) -> None:
    """Test Markdown formatter output."""
    formatter = MarkdownFormatter()
    output = StringIO()

    formatter.format(sample_report, output)
    result = output.getvalue()

    # Check header
    assert "# Dead Code Analysis Report" in result

    # Check entry points section
    assert "## Entry Points (2)" in result
    assert "### MAIN" in result
    assert "### TEST" in result
    assert "| Function | File | Line |" in result

    # Check findings section
    assert "## Dead Code Findings (3)" in result
    assert "### High Confidence" in result
    assert "### Medium Confidence" in result
    assert "### Low Confidence" in result

    # Check specific findings
    assert "`_legacy_converter`" in result
    assert "utils/converters.py:45-67" in result
    assert "`public_utility`" in result
    assert "`callback_handler`" in result

    # Check summary section
    assert "## Summary" in result
    assert "| Total Functions | 100 |" in result
    assert "| Reachable | 97 (97.0%) |" in result
    assert "| Unreachable | 3 (3.0%) |" in result


def test_get_formatter_console() -> None:
    """Test get_formatter factory with console format."""
    formatter = get_formatter("console")
    assert isinstance(formatter, ConsoleFormatter)


def test_get_formatter_json() -> None:
    """Test get_formatter factory with JSON format."""
    formatter = get_formatter("json")
    assert isinstance(formatter, JsonFormatter)


def test_get_formatter_sarif() -> None:
    """Test get_formatter factory with SARIF format."""
    formatter = get_formatter("sarif")
    assert isinstance(formatter, SarifFormatter)


def test_get_formatter_markdown() -> None:
    """Test get_formatter factory with Markdown format."""
    formatter = get_formatter("markdown")
    assert isinstance(formatter, MarkdownFormatter)


def test_get_formatter_case_insensitive() -> None:
    """Test get_formatter is case-insensitive."""
    formatter1 = get_formatter("JSON")
    formatter2 = get_formatter("Json")
    formatter3 = get_formatter("json")

    assert isinstance(formatter1, JsonFormatter)
    assert isinstance(formatter2, JsonFormatter)
    assert isinstance(formatter3, JsonFormatter)


def test_get_formatter_invalid() -> None:
    """Test get_formatter raises error for invalid format."""
    with pytest.raises(ValueError, match="Invalid format: invalid"):
        get_formatter("invalid")


def test_empty_report_console() -> None:
    """Test console formatter with empty report."""
    report = DeadCodeReport(
        entry_points=[],
        findings=[],
        total_functions=10,
        reachable_count=10,
        unreachable_count=0,
    )

    formatter = ConsoleFormatter()
    output = StringIO()
    formatter.format(report, output)
    result = output.getvalue()

    assert "Entry Points Detected:" in result
    assert "0" in result
    assert "Dead Code Findings:" in result
    assert "Total Functions" in result
    assert "10" in result


def test_empty_report_json() -> None:
    """Test JSON formatter with empty report."""
    report = DeadCodeReport(
        entry_points=[],
        findings=[],
        total_functions=10,
        reachable_count=10,
        unreachable_count=0,
    )

    formatter = JsonFormatter()
    output = StringIO()
    formatter.format(report, output)
    result = output.getvalue()

    data = json.loads(result)
    assert len(data["entry_points"]) == 0
    assert len(data["findings"]) == 0
    assert data["summary"]["total_functions"] == 10
    assert data["summary"]["unreachable"] == 0


def test_sarif_includes_caveats(sample_report: DeadCodeReport) -> None:
    """Test SARIF formatter includes caveats as notes."""
    formatter = SarifFormatter()
    output = StringIO()
    formatter.format(sample_report, output)

    data = json.loads(output.getvalue())
    results = data["runs"][0]["results"]

    # Find LOW confidence finding (has caveats)
    low_result = [r for r in results if r["ruleId"] == "dead-code-low"][0]
    assert "notes" in low_result
    assert len(low_result["notes"]) > 0
    assert "dynamically registered" in low_result["notes"][0]["text"]


def test_console_formatter_truncates_long_lists(sample_report: DeadCodeReport) -> None:
    """Test console formatter truncates long lists of findings."""
    # Create report with many findings
    findings = []
    for i in range(15):
        findings.append(
            DeadCodeFinding(
                function_name=f"func_{i}",
                file_path=f"file_{i}.py",
                start_line=i,
                end_line=i + 10,
                confidence=Confidence.HIGH,
                reason="Test finding",
                caveats=[],
            )
        )

    report = DeadCodeReport(
        entry_points=[],
        findings=findings,
        total_functions=100,
        reachable_count=85,
        unreachable_count=15,
    )

    formatter = ConsoleFormatter()
    output = StringIO()
    formatter.format(report, output)
    result = output.getvalue()

    # Should show "and X more" message
    assert "more high confidence findings" in result
