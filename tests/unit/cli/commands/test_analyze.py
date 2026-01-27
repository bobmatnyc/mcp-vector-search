"""Tests for the analyze CLI command."""

import pytest
from typer.testing import CliRunner

from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics
from mcp_vector_search.cli.commands.analyze import (
    _analyze_file,
    _find_analyzable_files,
    analyze_app,
)

runner = CliRunner()


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_help(self):
        """Test that analyze --help works."""
        result = runner.invoke(analyze_app, ["--help"])
        assert result.exit_code == 0
        assert "Analyze code complexity" in result.stdout

    def test_analyze_complexity_help(self):
        """Test that analyze complexity --help works."""
        result = runner.invoke(analyze_app, ["complexity", "--help"])
        assert result.exit_code == 0
        assert "complexity" in result.stdout.lower()

    @pytest.mark.asyncio
    async def test_find_analyzable_files_no_filter(self, tmp_path):
        """Test finding files without filters."""
        # Create test files
        (tmp_path / "test.py").write_text("def foo(): pass")
        (tmp_path / "test.js").write_text("function bar() {}")
        (tmp_path / "README.md").write_text("# Test")

        from mcp_vector_search.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        files = _find_analyzable_files(tmp_path, None, None, registry)

        # Should find Python, JavaScript, and Markdown files (TextParser handles .md)
        assert len(files) >= 2
        assert any(f.name == "test.py" for f in files)
        assert any(f.name == "test.js" for f in files)
        # Markdown files are also found because TextParser supports them
        assert any(f.name == "README.md" for f in files)

    @pytest.mark.asyncio
    async def test_find_analyzable_files_with_language_filter(self, tmp_path):
        """Test finding files with language filter."""
        # Create test files
        (tmp_path / "test.py").write_text("def foo(): pass")
        (tmp_path / "test.js").write_text("function bar() {}")

        from mcp_vector_search.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        files = _find_analyzable_files(tmp_path, "python", None, registry)

        # Should find only Python files
        assert len(files) == 1
        assert files[0].name == "test.py"

    @pytest.mark.asyncio
    async def test_find_analyzable_files_with_path_filter(self, tmp_path):
        """Test finding files with path filter."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        from mcp_vector_search.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        files = _find_analyzable_files(tmp_path, None, test_file, registry)

        # Should find only the specified file
        assert len(files) == 1
        assert files[0] == test_file

    @pytest.mark.asyncio
    async def test_find_analyzable_files_ignores_directories(self, tmp_path):
        """Test that common directories are ignored."""
        # Create ignored directories
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "test.js").write_text("function bar() {}")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("config")
        (tmp_path / "test.py").write_text("def foo(): pass")

        from mcp_vector_search.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        files = _find_analyzable_files(tmp_path, None, None, registry)

        # Should not find files in ignored directories
        assert len(files) == 1
        assert files[0].name == "test.py"

    @pytest.mark.asyncio
    async def test_analyze_file_basic(self, tmp_path):
        """Test analyzing a simple file."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def simple_function():
    return 42

def complex_function(a, b, c):
    if a > 0:
        for i in range(b):
            if i % 2 == 0:
                print(i)
    return a + b + c
"""
        )

        from mcp_vector_search.analysis import (
            CognitiveComplexityCollector,
            CyclomaticComplexityCollector,
        )
        from mcp_vector_search.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        collectors = [CognitiveComplexityCollector(), CyclomaticComplexityCollector()]

        file_metrics = await _analyze_file(test_file, registry, collectors)

        assert file_metrics is not None
        assert file_metrics.file_path == str(test_file)
        assert file_metrics.function_count >= 2
        assert len(file_metrics.chunks) >= 2

    @pytest.mark.asyncio
    async def test_analyze_file_empty(self, tmp_path):
        """Test analyzing an empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        from mcp_vector_search.parsers.registry import ParserRegistry

        registry = ParserRegistry()
        collectors = []

        file_metrics = await _analyze_file(test_file, registry, collectors)

        # Empty files should return None
        assert file_metrics is None

    def test_analyze_command_json_output(self, tmp_path):
        """Test analyze complexity command with JSON output."""
        # Create a simple test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        result = runner.invoke(
            analyze_app,
            ["complexity", "--project-root", str(tmp_path), "--json", "--quick"],
            catch_exceptions=False,
        )

        # Should succeed (exit code 0)
        assert result.exit_code == 0

        # Should contain JSON-like output with key fields
        # Rich's print_json adds formatting/colors, so we check for key presence
        assert "project_root" in result.stdout
        assert "analyzed_at" in result.stdout
        assert "total_files" in result.stdout
        assert "complexity_distribution" in result.stdout

    def test_analyze_command_quick_mode(self, tmp_path):
        """Test analyze complexity command in quick mode."""
        # Create a simple test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        result = runner.invoke(
            analyze_app,
            ["complexity", "--project-root", str(tmp_path), "--quick"],
        )

        # Should succeed
        assert result.exit_code == 0 or "No files found" in result.stdout

    def test_analyze_command_language_filter(self, tmp_path):
        """Test analyze complexity command with language filter."""
        # Create test files
        (tmp_path / "test.py").write_text("def foo(): pass")
        (tmp_path / "test.js").write_text("function bar() {}")

        result = runner.invoke(
            analyze_app,
            [
                "complexity",
                "--project-root",
                str(tmp_path),
                "--language",
                "python",
                "--quick",
            ],
        )

        # Should succeed
        assert result.exit_code == 0 or "No files found" in result.stdout


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_print_summary(self):
        """Test printing project summary."""
        from mcp_vector_search.analysis.reporters.console import ConsoleReporter

        metrics = ProjectMetrics(project_root="/test")
        metrics.total_files = 10
        metrics.total_lines = 1000
        metrics.total_functions = 50
        metrics.total_classes = 5
        metrics.avg_file_complexity = 7.5

        reporter = ConsoleReporter()
        # Should not raise any exceptions
        reporter.print_summary(metrics)

    def test_print_distribution(self):
        """Test printing grade distribution."""
        from mcp_vector_search.analysis.reporters.console import ConsoleReporter

        metrics = ProjectMetrics(project_root="/test")

        # Create some file metrics with chunks
        file_metrics = FileMetrics(file_path="test.py")
        file_metrics.chunks = [
            ChunkMetrics(cognitive_complexity=3),  # Grade A
            ChunkMetrics(cognitive_complexity=8),  # Grade B
            ChunkMetrics(cognitive_complexity=15),  # Grade C
        ]
        metrics.files["test.py"] = file_metrics

        reporter = ConsoleReporter()
        # Should not raise any exceptions
        reporter.print_distribution(metrics)

    def test_print_hotspots(self):
        """Test printing complexity hotspots."""
        from mcp_vector_search.analysis.reporters.console import ConsoleReporter

        metrics = ProjectMetrics(project_root="/test")

        # Create file metrics with different complexity
        file1 = FileMetrics(file_path="complex.py")
        file1.chunks = [ChunkMetrics(cognitive_complexity=25)]
        file1.compute_aggregates()

        file2 = FileMetrics(file_path="simple.py")
        file2.chunks = [ChunkMetrics(cognitive_complexity=3)]
        file2.compute_aggregates()

        metrics.files["complex.py"] = file1
        metrics.files["simple.py"] = file2

        reporter = ConsoleReporter()
        # Should not raise any exceptions
        reporter.print_hotspots(metrics, top=5)

    def test_print_recommendations(self):
        """Test printing recommendations."""
        from mcp_vector_search.analysis.reporters.console import ConsoleReporter

        metrics = ProjectMetrics(project_root="/test")

        # Create file with poor health
        file_metrics = FileMetrics(file_path="bad.py")
        file_metrics.chunks = [
            ChunkMetrics(cognitive_complexity=35),  # Grade F
            ChunkMetrics(cognitive_complexity=28),  # Grade D
        ]
        file_metrics.compute_aggregates()
        metrics.files["bad.py"] = file_metrics

        metrics.compute_aggregates()

        reporter = ConsoleReporter()
        # Should not raise any exceptions
        reporter.print_recommendations(metrics)
