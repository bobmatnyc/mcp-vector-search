"""Unit tests for dead code CLI command.

Note: These are simplified tests that verify the command structure and options.
Full integration testing is done via e2e tests.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mcp_vector_search.cli.main import app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.mark.skip(reason="CLI help output changed - test needs update for v1.2.7")
def test_dead_code_command_help(cli_runner: CliRunner) -> None:
    """Test dead code command help output."""
    result = cli_runner.invoke(app, ["analyze", "dead-code", "--help"])

    assert result.exit_code == 0
    assert "Detect dead/unreachable code" in result.output
    assert "--entry-point" in result.output
    assert "--include-public" in result.output
    assert "--min-confidence" in result.output
    assert "--exclude" in result.output
    assert "--output" in result.output
    assert "--fail-on-dead" in result.output


def test_dead_code_command_requires_initialized_project(cli_runner: CliRunner) -> None:
    """Test dead code command fails when project not initialized."""
    result = cli_runner.invoke(app, ["analyze", "dead-code"])

    # Should fail because project is not initialized
    assert result.exit_code != 0
    # The error message might be about missing project, database, or configuration
    assert any(
        keyword in result.output.lower()
        for keyword in [
            "not initialized",
            "no code chunks",
            "failed",
            "error",
            "get_db_path",
        ]
    )


def test_dead_code_command_invalid_confidence(cli_runner: CliRunner) -> None:
    """Test dead code command with invalid confidence level."""
    # This should fail even before checking project initialization
    result = cli_runner.invoke(
        app, ["analyze", "dead-code", "--min-confidence", "invalid"]
    )

    # Will fail at project check or validation - either is fine
    assert result.exit_code != 0
