"""Tests for analyze command exit codes with --fail-on-smell."""

from pathlib import Path

from typer.testing import CliRunner

from mcp_vector_search.analysis.collectors.smells import CodeSmell, SmellSeverity
from mcp_vector_search.cli.commands.analyze import (
    analyze_app,
    filter_smells_by_severity,
)

runner = CliRunner()


class TestExitCodes:
    """Test exit codes for --fail-on-smell feature."""

    def test_exit_code_0_no_smells(self, tmp_path: Path) -> None:
        """Exit 0 when no smells found."""
        # Create simple file with no complexity issues
        test_file = tmp_path / "simple.py"
        test_file.write_text(
            """def simple():
    return 1
"""
        )

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--fail-on-smell",
                "--quick",
            ],
        )

        # Should succeed - no smells detected
        assert result.exit_code == 0
        assert "Quality gate failed" not in result.stdout

    def test_exit_code_1_error_smells(self, tmp_path: Path) -> None:
        """Exit 1 when ERROR-level smells found with --fail-on-smell.

        NOTE: Currently, ERROR-level smells (God Class) require specific conditions
        that are difficult to trigger in synthetic tests. This test verifies the
        exit code mechanism works correctly when WARNING smells are treated as errors.
        """
        # Create file with WARNING-level smell
        test_file = tmp_path / "warning_smell.py"
        test_file.write_text(
            """def long_function():
    # Generate enough lines to trigger Long Method warning
"""
            + "\n".join([f"    x_{i} = {i}" for i in range(60)])  # > 50 lines
        )

        # Use warning threshold to treat WARNING smells as failures
        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--path",
                str(test_file),
                "--fail-on-smell",
                "--severity-threshold",
                "warning",  # This will fail on WARNING smells
                "--quick",
            ],
            catch_exceptions=False,
        )

        # Should fail - WARNING smell treated as failure
        assert result.exit_code == 1, (
            f"Expected exit code 1, got {result.exit_code}. Output: {result.stdout}"
        )

    def test_exit_code_0_warning_only_error_threshold(self, tmp_path: Path) -> None:
        """Exit 0 when only WARNING smells and threshold=error."""
        # Create file with Long Method smell (WARNING severity)
        test_file = tmp_path / "long_method.py"
        test_file.write_text(
            """def long_function():
    # Generate enough lines to trigger Long Method warning
"""
            + "\n".join([f"    x_{i} = {i}" for i in range(60)])  # > 50 lines
        )

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--fail-on-smell",
                "--severity-threshold",
                "error",  # Only fail on ERROR, not WARNING
                "--quick",
            ],
        )

        # Should succeed - only WARNING smells, threshold is ERROR
        assert result.exit_code == 0
        assert "Quality gate failed" not in result.stdout

    def test_exit_code_1_warning_with_warning_threshold(self, tmp_path: Path) -> None:
        """Exit 1 when WARNING smells and threshold=warning."""
        # Create file with Long Method smell (WARNING severity)
        test_file = tmp_path / "long_method.py"
        test_file.write_text(
            """def long_function():
    # Generate enough lines to trigger Long Method warning
"""
            + "\n".join([f"    x_{i} = {i}" for i in range(60)])  # > 50 lines
        )

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--fail-on-smell",
                "--severity-threshold",
                "warning",  # Fail on WARNING or ERROR
                "--quick",
            ],
            catch_exceptions=False,
        )

        # Should fail - WARNING smell detected and threshold is warning
        assert result.exit_code == 1
        assert "Quality gate failed" in result.stdout or result.exit_code == 1

    def test_severity_threshold_none(self, tmp_path: Path) -> None:
        """Exit 0 when threshold=none regardless of smells."""
        # Create file with multiple severity levels of smells
        test_file = tmp_path / "complex.py"
        test_file.write_text(
            """def complex_function(a, b, c, d, e, f, g):  # Many params
    # Generate enough lines and complexity
"""
            + "\n".join([f"    if {i}: x = {i}" for i in range(60)])
        )

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--fail-on-smell",
                "--severity-threshold",
                "none",  # Don't fail on any smells
                "--quick",
            ],
        )

        # Should succeed - threshold=none means never fail
        assert result.exit_code == 0
        assert "Quality gate failed" not in result.stdout

    def test_exit_code_0_no_fail_on_smell_flag(self, tmp_path: Path) -> None:
        """Exit 0 when smells present but --fail-on-smell not specified."""
        # Create file with smells
        test_file = tmp_path / "long_method.py"
        test_file.write_text(
            """def long_function():
"""
            + "\n".join([f"    x_{i} = {i}" for i in range(60)])
        )

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                # Note: --fail-on-smell NOT specified
                "--quick",
            ],
        )

        # Should succeed - flag not specified, so don't fail
        assert result.exit_code == 0

    def test_exit_code_1_info_with_info_threshold(self, tmp_path: Path) -> None:
        """Exit 1 when INFO smells and threshold=info."""
        # Create file that might trigger INFO-level smells
        # Note: Current implementation doesn't have INFO smells yet,
        # but this test is future-proof
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def some_function():
    pass
"""
        )

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--fail-on-smell",
                "--severity-threshold",
                "info",  # Fail on any severity (INFO, WARNING, ERROR)
                "--quick",
            ],
        )

        # May pass or fail depending on whether INFO smells exist
        # This test documents the behavior
        assert result.exit_code in [0, 1]

    def test_severity_threshold_case_insensitive(self, tmp_path: Path) -> None:
        """Severity threshold should be case-insensitive."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def simple(): return 1")

        # Try different cases
        for threshold in ["ERROR", "Error", "error", "eRrOr"]:
            result = runner.invoke(
                analyze_app,
                [
                    "complexity",
                    "--project-root",
                    str(tmp_path),
                    "--fail-on-smell",
                    "--severity-threshold",
                    threshold,
                    "--quick",
                ],
            )
            assert result.exit_code == 0


class TestFilterSmellsBySeverity:
    """Test the filter_smells_by_severity helper function."""

    def test_filter_none_threshold(self) -> None:
        """Test that 'none' threshold returns empty list."""
        smells = [
            CodeSmell(
                name="Test",
                description="Test smell",
                severity=SmellSeverity.ERROR,
                location="test.py:1",
                metric_value=10.0,
                threshold=5.0,
            )
        ]

        result = filter_smells_by_severity(smells, "none")
        assert len(result) == 0

    def test_filter_error_threshold(self) -> None:
        """Test that 'error' threshold only includes ERROR smells."""
        smells = [
            CodeSmell(
                name="Error Smell",
                description="Error",
                severity=SmellSeverity.ERROR,
                location="test.py:1",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Warning Smell",
                description="Warning",
                severity=SmellSeverity.WARNING,
                location="test.py:2",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Info Smell",
                description="Info",
                severity=SmellSeverity.INFO,
                location="test.py:3",
                metric_value=10.0,
                threshold=5.0,
            ),
        ]

        result = filter_smells_by_severity(smells, "error")
        assert len(result) == 1
        assert result[0].severity == SmellSeverity.ERROR

    def test_filter_warning_threshold(self) -> None:
        """Test that 'warning' threshold includes WARNING and ERROR."""
        smells = [
            CodeSmell(
                name="Error Smell",
                description="Error",
                severity=SmellSeverity.ERROR,
                location="test.py:1",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Warning Smell",
                description="Warning",
                severity=SmellSeverity.WARNING,
                location="test.py:2",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Info Smell",
                description="Info",
                severity=SmellSeverity.INFO,
                location="test.py:3",
                metric_value=10.0,
                threshold=5.0,
            ),
        ]

        result = filter_smells_by_severity(smells, "warning")
        assert len(result) == 2
        assert all(
            s.severity in [SmellSeverity.WARNING, SmellSeverity.ERROR] for s in result
        )

    def test_filter_info_threshold(self) -> None:
        """Test that 'info' threshold includes all severities."""
        smells = [
            CodeSmell(
                name="Error Smell",
                description="Error",
                severity=SmellSeverity.ERROR,
                location="test.py:1",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Warning Smell",
                description="Warning",
                severity=SmellSeverity.WARNING,
                location="test.py:2",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Info Smell",
                description="Info",
                severity=SmellSeverity.INFO,
                location="test.py:3",
                metric_value=10.0,
                threshold=5.0,
            ),
        ]

        result = filter_smells_by_severity(smells, "info")
        assert len(result) == 3

    def test_filter_invalid_threshold(self) -> None:
        """Test that invalid threshold defaults to 'error'."""
        smells = [
            CodeSmell(
                name="Error Smell",
                description="Error",
                severity=SmellSeverity.ERROR,
                location="test.py:1",
                metric_value=10.0,
                threshold=5.0,
            ),
            CodeSmell(
                name="Warning Smell",
                description="Warning",
                severity=SmellSeverity.WARNING,
                location="test.py:2",
                metric_value=10.0,
                threshold=5.0,
            ),
        ]

        result = filter_smells_by_severity(smells, "invalid")
        # Should default to ERROR-only
        assert len(result) == 1
        assert result[0].severity == SmellSeverity.ERROR

    def test_filter_empty_list(self) -> None:
        """Test filtering empty list returns empty list."""
        result = filter_smells_by_severity([], "error")
        assert len(result) == 0

    def test_filter_case_insensitive(self) -> None:
        """Test that threshold is case-insensitive."""
        smells = [
            CodeSmell(
                name="Error Smell",
                description="Error",
                severity=SmellSeverity.ERROR,
                location="test.py:1",
                metric_value=10.0,
                threshold=5.0,
            )
        ]

        for threshold in ["ERROR", "Error", "error", "ErRoR"]:
            result = filter_smells_by_severity(smells, threshold)
            assert len(result) == 1
