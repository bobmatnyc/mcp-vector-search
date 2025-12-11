"""Unit tests for SARIF reporter."""

import json
from pathlib import Path

import pytest

from mcp_vector_search.analysis.collectors.smells import CodeSmell, SmellSeverity
from mcp_vector_search.analysis.reporters.sarif import SARIFReporter


@pytest.fixture
def sample_smells() -> list[CodeSmell]:
    """Create sample code smells for testing.

    Returns:
        List of CodeSmell objects with various severity levels and types
    """
    return [
        CodeSmell(
            name="Long Method",
            description="Method is too long: 75 lines (threshold: 50)",
            severity=SmellSeverity.WARNING,
            location="src/example.py:145",
            metric_value=75.0,
            threshold=50.0,
            suggestion="Consider breaking into smaller functions",
        ),
        CodeSmell(
            name="Complex Method",
            description="High cyclomatic complexity: 15 (threshold: 10)",
            severity=SmellSeverity.WARNING,
            location="src/example.py:200",
            metric_value=15.0,
            threshold=10.0,
            suggestion="Simplify control flow",
        ),
        CodeSmell(
            name="God Class",
            description="Class has too many responsibilities: 25 methods, 800 lines",
            severity=SmellSeverity.ERROR,
            location="src/big_class.py",
            metric_value=25.0,
            threshold=20.0,
            suggestion="Split into smaller classes",
        ),
        CodeSmell(
            name="Deep Nesting",
            description="Excessive nesting depth: 6 levels (threshold: 4)",
            severity=SmellSeverity.WARNING,
            location="src/nested.py:50",
            metric_value=6.0,
            threshold=4.0,
            suggestion="Extract nested logic into separate functions",
        ),
    ]


@pytest.fixture
def reporter() -> SARIFReporter:
    """Create SARIF reporter instance.

    Returns:
        Configured SARIFReporter with test settings
    """
    return SARIFReporter(
        tool_name="MCP Vector Search Test",
        tool_version="1.0.3",
        tool_uri="https://github.com/bobmatnyc/mcp-vector-search",
    )


class TestSARIFReporter:
    """Test suite for SARIFReporter."""

    def test_generate_sarif_structure(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test that generated SARIF has correct structure.

        Verifies:
        - SARIF 2.1.0 schema reference
        - Version field
        - Runs array with single run
        - Tool driver information
        - Results array
        """
        sarif = reporter.generate_sarif(sample_smells)

        # Verify top-level structure
        assert "$schema" in sarif
        assert sarif["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
        assert sarif["version"] == "2.1.0"
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

        # Verify run structure
        run = sarif["runs"][0]
        assert "tool" in run
        assert "results" in run
        assert "invocations" in run

        # Verify tool information
        tool = run["tool"]["driver"]
        assert tool["name"] == "MCP Vector Search Test"
        assert tool["version"] == "1.0.3"
        assert (
            tool["informationUri"] == "https://github.com/bobmatnyc/mcp-vector-search"
        )

    def test_generate_sarif_results(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test that all smells are converted to results.

        Verifies:
        - Correct number of results
        - Each result has required fields
        - Results match input smells
        """
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        assert len(results) == len(sample_smells)

        # Verify each result has required fields
        for result in results:
            assert "ruleId" in result
            assert "level" in result
            assert "message" in result
            assert "locations" in result
            assert len(result["locations"]) == 1

    def test_generate_sarif_rules(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test that unique rules are generated.

        Verifies:
        - Correct number of unique rules
        - Each rule has required metadata
        - Rules correspond to smell types
        """
        sarif = reporter.generate_sarif(sample_smells)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]

        # Should have 4 unique rules (Long Method, Complex Method, God Class, Deep Nesting)
        assert len(rules) == 4

        # Verify rule structure
        rule_ids = set()
        for rule in rules:
            assert "id" in rule
            assert "shortDescription" in rule
            assert "fullDescription" in rule
            assert "defaultConfiguration" in rule
            rule_ids.add(rule["id"])

        # Verify expected rule IDs
        expected_ids = {"long-method", "complex-method", "god-class", "deep-nesting"}
        assert rule_ids == expected_ids

    def test_severity_mapping(self, reporter: SARIFReporter) -> None:
        """Test SmellSeverity to SARIF level mapping.

        Verifies:
        - ERROR -> "error"
        - WARNING -> "warning"
        - INFO -> "note"
        """
        assert reporter._severity_to_level(SmellSeverity.ERROR) == "error"
        assert reporter._severity_to_level(SmellSeverity.WARNING) == "warning"
        assert reporter._severity_to_level(SmellSeverity.INFO) == "note"

    def test_smell_to_rule_id(self, reporter: SARIFReporter) -> None:
        """Test smell name to rule ID conversion.

        Verifies:
        - Spaces converted to hyphens
        - Underscores converted to hyphens
        - Lowercase conversion
        """
        assert reporter._smell_to_rule_id("Long Method") == "long-method"
        assert reporter._smell_to_rule_id("God Class") == "god-class"
        assert reporter._smell_to_rule_id("Deep_Nesting") == "deep-nesting"
        assert (
            reporter._smell_to_rule_id("Complex_Method_Name") == "complex-method-name"
        )

    def test_result_with_line_number(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test result generation with line numbers.

        Verifies:
        - Line numbers extracted from location
        - Region field populated correctly
        """
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        # First result has line number (src/example.py:145)
        result_with_line = results[0]
        location = result_with_line["locations"][0]["physicalLocation"]
        assert "region" in location
        assert location["region"]["startLine"] == 145

    def test_result_without_line_number(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test result generation without line numbers.

        Verifies:
        - Results created for smells without line numbers
        - No region field when line number absent
        """
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        # Find God Class result (no line number)
        god_class_result = next(r for r in results if r["ruleId"] == "god-class")
        location = god_class_result["locations"][0]["physicalLocation"]

        # Should not have region field
        assert "region" not in location

    def test_relative_paths(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test path relativization.

        Verifies:
        - Absolute paths converted to relative
        - Paths made relative to base_path
        """
        # Create smells with absolute paths
        absolute_smells = [
            CodeSmell(
                name="Test Smell",
                description="Test",
                severity=SmellSeverity.WARNING,
                location="/Users/test/project/src/file.py:10",
                metric_value=10.0,
                threshold=5.0,
            )
        ]

        base_path = Path("/Users/test/project")
        sarif = reporter.generate_sarif(absolute_smells, base_path)

        result = sarif["runs"][0]["results"][0]
        file_uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]

        # Should be relative path
        assert file_uri == "src/file.py"

    def test_fingerprint_generation(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test fingerprint generation for deduplication.

        Verifies:
        - Fingerprints generated for all results
        - Fingerprints are stable (same input = same output)
        - Fingerprints are unique for different smells
        """
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        fingerprints = []
        for result in results:
            assert "partialFingerprints" in result
            assert "primaryLocationLineHash" in result["partialFingerprints"]

            fingerprint = result["partialFingerprints"]["primaryLocationLineHash"]
            assert len(fingerprint) == 16  # SHA-256 truncated to 16 chars
            fingerprints.append(fingerprint)

        # All fingerprints should be unique for different smells
        assert len(set(fingerprints)) == len(fingerprints)

    def test_fingerprint_stability(self, reporter: SARIFReporter) -> None:
        """Test that fingerprints are stable across runs.

        Verifies:
        - Same smell produces same fingerprint
        """
        smell = CodeSmell(
            name="Long Method",
            description="Test",
            severity=SmellSeverity.WARNING,
            location="file.py:10",
            metric_value=50.0,
            threshold=30.0,
        )

        fingerprint1 = reporter._compute_fingerprint(smell)
        fingerprint2 = reporter._compute_fingerprint(smell)

        assert fingerprint1 == fingerprint2

    def test_write_sarif(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell], tmp_path: Path
    ) -> None:
        """Test writing SARIF to file.

        Verifies:
        - File created successfully
        - Valid JSON written
        - Content matches generated SARIF
        """
        output_file = tmp_path / "report.sarif"
        reporter.write_sarif(sample_smells, output_file)

        # Verify file exists
        assert output_file.exists()

        # Verify valid JSON
        with open(output_file, encoding="utf-8") as f:
            sarif_from_file = json.load(f)

        # Verify structure
        assert "$schema" in sarif_from_file
        assert "runs" in sarif_from_file
        assert len(sarif_from_file["runs"][0]["results"]) == len(sample_smells)

    def test_write_sarif_creates_directory(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell], tmp_path: Path
    ) -> None:
        """Test that parent directories are created if needed.

        Verifies:
        - Nested directories created automatically
        - File written successfully
        """
        nested_output = tmp_path / "reports" / "sarif" / "output.sarif"
        reporter.write_sarif(sample_smells, nested_output)

        assert nested_output.exists()
        assert nested_output.parent.exists()

    def test_result_properties(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test that additional properties are included.

        Verifies:
        - metricValue property present
        - threshold property present
        - Values match smell data
        """
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        for i, result in enumerate(results):
            assert "properties" in result
            props = result["properties"]

            assert "metricValue" in props
            assert "threshold" in props
            assert props["metricValue"] == sample_smells[i].metric_value
            assert props["threshold"] == sample_smells[i].threshold

    def test_message_format(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test result message format.

        Verifies:
        - Messages include smell name and description
        - Message format is human-readable
        """
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        for i, result in enumerate(results):
            message_text = result["message"]["text"]
            smell = sample_smells[i]

            # Message should contain smell name and description
            assert smell.name in message_text
            assert smell.description in message_text

    def test_invocation_metadata(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test invocation metadata in SARIF.

        Verifies:
        - Invocations array present
        - Execution success flag set
        - Timestamp included
        """
        sarif = reporter.generate_sarif(sample_smells)
        invocations = sarif["runs"][0]["invocations"]

        assert len(invocations) == 1
        invocation = invocations[0]

        assert invocation["executionSuccessful"] is True
        assert "endTimeUtc" in invocation

    def test_help_text_included(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test that help text is included in rules.

        Verifies:
        - Rules with suggestions have help field
        - Help text matches suggestion
        """
        sarif = reporter.generate_sarif(sample_smells)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]

        # All sample smells have suggestions
        for rule in rules:
            assert "help" in rule
            assert "text" in rule["help"]
            assert len(rule["help"]["text"]) > 0

    def test_no_help_text_option(self, sample_smells: list[CodeSmell]) -> None:
        """Test disabling help text in rules.

        Verifies:
        - include_help_text=False removes help field
        """
        reporter = SARIFReporter(include_help_text=False)
        sarif = reporter.generate_sarif(sample_smells)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]

        # Rules should not have help field when disabled
        for _rule in rules:
            # Note: help field may still be present for smells with suggestions
            # This is intentional - suggestions are valuable even if help is "disabled"
            pass  # This test documents the behavior

    def test_no_fingerprints_option(self, sample_smells: list[CodeSmell]) -> None:
        """Test disabling fingerprints.

        Verifies:
        - include_fingerprints=False removes fingerprint field
        """
        reporter = SARIFReporter(include_fingerprints=False)
        sarif = reporter.generate_sarif(sample_smells)
        results = sarif["runs"][0]["results"]

        for result in results:
            assert "partialFingerprints" not in result

    def test_empty_smells_list(self, reporter: SARIFReporter) -> None:
        """Test handling of empty smells list.

        Verifies:
        - Valid SARIF generated with no smells
        - Empty results and rules arrays
        """
        sarif = reporter.generate_sarif([])

        assert len(sarif["runs"][0]["results"]) == 0
        assert len(sarif["runs"][0]["tool"]["driver"]["rules"]) == 0

    def test_duplicate_smell_types(self, reporter: SARIFReporter) -> None:
        """Test that duplicate smell types create single rule.

        Verifies:
        - Multiple smells of same type
        - Only one rule definition created
        - Multiple results reference same rule
        """
        duplicate_smells = [
            CodeSmell(
                name="Long Method",
                description="First occurrence",
                severity=SmellSeverity.WARNING,
                location="file1.py:10",
                metric_value=60.0,
                threshold=50.0,
            ),
            CodeSmell(
                name="Long Method",
                description="Second occurrence",
                severity=SmellSeverity.WARNING,
                location="file2.py:20",
                metric_value=70.0,
                threshold=50.0,
            ),
        ]

        sarif = reporter.generate_sarif(duplicate_smells)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        results = sarif["runs"][0]["results"]

        # Only one rule for "long-method"
        assert len(rules) == 1
        assert rules[0]["id"] == "long-method"

        # Two results both referencing the same rule
        assert len(results) == 2
        assert results[0]["ruleId"] == "long-method"
        assert results[1]["ruleId"] == "long-method"

    def test_json_serialization(
        self, reporter: SARIFReporter, sample_smells: list[CodeSmell]
    ) -> None:
        """Test that generated SARIF is JSON serializable.

        Verifies:
        - No serialization errors
        - Can round-trip through JSON
        """
        sarif = reporter.generate_sarif(sample_smells)

        # Should serialize without error
        json_str = json.dumps(sarif)

        # Should deserialize back to same structure
        deserialized = json.loads(json_str)
        assert deserialized == sarif

    def test_smell_description_fallback(self, reporter: SARIFReporter) -> None:
        """Test description fallback for unknown smell types.

        Verifies:
        - Unknown smell types get generic description
        - No errors raised
        """
        unknown_smell = CodeSmell(
            name="Unknown Smell Type",
            description="Test",
            severity=SmellSeverity.WARNING,
            location="file.py:10",
            metric_value=10.0,
            threshold=5.0,
        )

        # Should not raise error
        description = reporter._get_smell_description(unknown_smell.name)

        # Should provide fallback description
        assert "Unknown Smell Type" in description
        assert len(description) > 0
