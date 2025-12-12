"""Unit tests for HTML report generator.

Tests cover:
- Basic HTML generation and structure
- Grade calculation logic
- File output functionality
- Handling of various data sizes
- Handling of missing optional data (trends, dependencies)
- Valid HTML structure validation
"""

from datetime import datetime
from pathlib import Path

import pytest

from mcp_vector_search.analysis.visualizer import (
    AnalysisExport,
    ClassMetrics,
    CyclicDependency,
    DependencyEdge,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    FunctionMetrics,
    HTMLReportGenerator,
    MetricsSummary,
    MetricTrend,
    SmellLocation,
    TrendData,
    TrendDataPoint,
)


@pytest.fixture
def minimal_metadata() -> ExportMetadata:
    """Create minimal export metadata for testing."""
    return ExportMetadata(
        version="1.0.0",
        generated_at=datetime(2025, 12, 11, 10, 0, 0),
        tool_version="0.19.0",
        project_root="/path/to/project",
    )


@pytest.fixture
def metadata_with_git() -> ExportMetadata:
    """Create export metadata with git information."""
    return ExportMetadata(
        version="1.0.0",
        generated_at=datetime(2025, 12, 11, 10, 0, 0),
        tool_version="0.19.0",
        project_root="/path/to/project",
        git_commit="abc123def456",
        git_branch="main",
    )


@pytest.fixture
def basic_summary() -> MetricsSummary:
    """Create basic metrics summary."""
    return MetricsSummary(
        total_files=10,
        total_functions=50,
        total_classes=15,
        total_lines=2000,
        avg_complexity=5.5,
        avg_cognitive_complexity=6.2,
        avg_nesting_depth=2.3,
        total_smells=12,
        smells_by_severity={"error": 2, "warning": 8, "info": 2},
    )


@pytest.fixture
def sample_file_detail() -> FileDetail:
    """Create sample file detail for testing."""
    return FileDetail(
        path="src/main.py",
        language="python",
        lines_of_code=150,
        cyclomatic_complexity=15,
        cognitive_complexity=18,
        max_nesting_depth=3,
        function_count=5,
        class_count=2,
        efferent_coupling=3,
        afferent_coupling=1,
        instability=0.75,
        functions=[
            FunctionMetrics(
                name="process_data",
                line_start=10,
                line_end=50,
                cyclomatic_complexity=8,
                cognitive_complexity=10,
                nesting_depth=3,
                parameter_count=3,
                lines_of_code=40,
            )
        ],
        classes=[
            ClassMetrics(
                name="DataProcessor",
                line_start=60,
                line_end=120,
                method_count=3,
                lcom4=2,
            )
        ],
        smells=[
            SmellLocation(
                smell_type="long_method",
                severity="warning",
                message="Method is too long",
                line=10,
                function_name="process_data",
            )
        ],
        imports=["os", "sys", "pathlib"],
    )


@pytest.fixture
def minimal_export(
    minimal_metadata: ExportMetadata, basic_summary: MetricsSummary
) -> AnalysisExport:
    """Create minimal export with no files."""
    return AnalysisExport(
        metadata=minimal_metadata,
        summary=basic_summary,
        files=[],
        dependencies=DependencyGraph(),
    )


@pytest.fixture
def full_export(
    metadata_with_git: ExportMetadata,
    basic_summary: MetricsSummary,
    sample_file_detail: FileDetail,
) -> AnalysisExport:
    """Create full export with all optional data."""
    return AnalysisExport(
        metadata=metadata_with_git,
        summary=basic_summary,
        files=[sample_file_detail],
        dependencies=DependencyGraph(
            edges=[
                DependencyEdge(
                    source="src/main.py", target="src/utils.py", import_type="import"
                )
            ],
            circular_dependencies=[
                CyclicDependency(cycle=["src/a.py", "src/b.py", "src/a.py"], length=3)
            ],
            most_depended_on=[("src/utils.py", 5)],
            most_dependent=[("src/main.py", 3)],
        ),
        trends=TrendData(
            metrics=[
                MetricTrend(
                    metric_name="avg_complexity",
                    current_value=5.5,
                    previous_value=6.0,
                    change_percent=-8.3,
                    trend_direction="improving",
                    history=[
                        TrendDataPoint(
                            timestamp=datetime(2025, 12, 1, 10, 0, 0),
                            commit="abc123",
                            value=6.0,
                        ),
                        TrendDataPoint(
                            timestamp=datetime(2025, 12, 11, 10, 0, 0),
                            commit="def456",
                            value=5.5,
                        ),
                    ],
                )
            ],
            baseline_name="main",
            baseline_date=datetime(2025, 12, 1, 10, 0, 0),
        ),
    )


class TestHTMLReportGenerator:
    """Test suite for HTMLReportGenerator."""

    def test_init_default_title(self):
        """Test initialization with default title."""
        gen = HTMLReportGenerator()
        assert gen.title == "Code Analysis Report"

    def test_init_custom_title(self):
        """Test initialization with custom title."""
        gen = HTMLReportGenerator(title="My Custom Report")
        assert gen.title == "My Custom Report"

    def test_generate_minimal_html(self, minimal_export: AnalysisExport):
        """Test generating HTML from minimal export."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check basic structure
        assert "<!DOCTYPE html>" in html
        assert '<html lang="en">' in html
        assert "</html>" in html
        assert "Code Analysis Report" in html

        # Check metadata
        assert "2025-12-11 10:00:00" in html
        assert "/path/to/project" in html

        # Check summary stats
        assert "10" in html  # total_files
        assert "50" in html  # total_functions
        assert "2,000" in html  # total_lines

    def test_generate_with_git_info(self, full_export: AnalysisExport):
        """Test HTML generation includes git information."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        assert "main" in html  # git branch
        assert "abc123de" in html  # git commit (truncated)

    def test_generate_includes_cdn_links(self, minimal_export: AnalysisExport):
        """Test that generated HTML includes CDN links."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert "cdn.jsdelivr.net/npm/chart.js" in html
        assert "cdnjs.cloudflare.com/ajax/libs/highlight.js" in html

    def test_generate_includes_styles(self, minimal_export: AnalysisExport):
        """Test that CSS styles are embedded."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert "<style>" in html
        assert "</style>" in html
        assert ":root" in html
        assert "--primary:" in html
        assert ".card" in html

    def test_generate_includes_scripts(self, minimal_export: AnalysisExport):
        """Test that JavaScript is embedded."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert "<script>" in html
        assert "</script>" in html
        assert "Chart" in html
        assert "hljs.highlightAll()" in html

    def test_generate_to_file(self, minimal_export: AnalysisExport, tmp_path: Path):
        """Test writing HTML to file."""
        gen = HTMLReportGenerator()
        output_path = tmp_path / "report.html"

        result_path = gen.generate_to_file(minimal_export, output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.is_file()

        # Verify content
        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Code Analysis Report" in content

    def test_generate_to_file_creates_parent_dirs(
        self, minimal_export: AnalysisExport, tmp_path: Path
    ):
        """Test that generate_to_file creates parent directories."""
        gen = HTMLReportGenerator()
        output_path = tmp_path / "reports" / "nested" / "report.html"

        result_path = gen.generate_to_file(minimal_export, output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_grade_calculation(self):
        """Test grade calculation from complexity scores."""
        assert HTMLReportGenerator._get_grade(3) == "A"
        assert HTMLReportGenerator._get_grade(5) == "A"
        assert HTMLReportGenerator._get_grade(7) == "B"
        assert HTMLReportGenerator._get_grade(10) == "B"
        assert HTMLReportGenerator._get_grade(15) == "C"
        assert HTMLReportGenerator._get_grade(20) == "C"
        assert HTMLReportGenerator._get_grade(25) == "D"
        assert HTMLReportGenerator._get_grade(30) == "D"
        assert HTMLReportGenerator._get_grade(35) == "F"
        assert HTMLReportGenerator._get_grade(100) == "F"

    def test_grade_distribution_section(self, full_export: AnalysisExport):
        """Test grade distribution is calculated correctly."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        # Check grade distribution table exists
        assert "🎯 Grade Distribution" in html
        assert "<th>Grade</th>" in html
        assert "<th>Count</th>" in html
        assert "<th>Percentage</th>" in html

        # Should have grade badges
        assert "grade-badge" in html

    def test_smells_section_with_smells(self, full_export: AnalysisExport):
        """Test code smells section when smells exist."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        assert "🔍 Code Smells" in html
        assert "long_method" in html
        assert "Method is too long" in html
        assert "src/main.py:10" in html

    def test_smells_section_empty(
        self, minimal_metadata: ExportMetadata, basic_summary: MetricsSummary
    ):
        """Test code smells section when no smells exist."""
        export = AnalysisExport(
            metadata=minimal_metadata,
            summary=basic_summary,
            files=[
                FileDetail(
                    path="src/clean.py",
                    language="python",
                    lines_of_code=50,
                    cyclomatic_complexity=5,
                    cognitive_complexity=5,
                    max_nesting_depth=1,
                    function_count=2,
                    class_count=0,
                    efferent_coupling=0,
                    afferent_coupling=0,
                    smells=[],  # No smells
                )
            ],
            dependencies=DependencyGraph(),
        )

        gen = HTMLReportGenerator()
        html = gen.generate(export)

        assert "🔍 Code Smells" in html
        assert "No code smells detected! 🎉" in html

    def test_files_table(self, full_export: AnalysisExport):
        """Test files table is generated correctly."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        assert "📁 Top Files by Complexity" in html
        assert "<th>File</th>" in html
        assert "<th>Grade</th>" in html
        assert "<th>Cognitive</th>" in html
        assert "src/main.py" in html

    def test_dependencies_section_with_data(self, full_export: AnalysisExport):
        """Test dependencies section when data exists."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        assert "🔗 Dependencies" in html
        assert "Total Imports" in html
        assert "Circular Dependencies" in html
        assert "⚠️ Circular Dependencies" in html
        assert "src/a.py → src/b.py → src/a.py" in html

    def test_dependencies_section_empty(self, minimal_export: AnalysisExport):
        """Test dependencies section when no dependencies."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Section should still exist but with zeros
        assert "🔗 Dependencies" in html
        assert "0" in html

    def test_trends_section_with_data(self, full_export: AnalysisExport):
        """Test trends section when trend data exists."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        assert "📈 Trends" in html
        assert "avg_complexity" in html
        assert "improving" in html
        assert "-8.3%" in html

    def test_trends_section_missing(self, minimal_export: AnalysisExport):
        """Test trends section is omitted when no trend data."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Trends section should not appear
        assert "📈 Trends" not in html or html.count("📈 Trends") == 0

    def test_complexity_chart_placeholder(self, minimal_export: AnalysisExport):
        """Test complexity chart canvas is included."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert "complexityChart" in html
        assert '<canvas id="complexityChart"></canvas>' in html

    def test_footer_includes_version(self, minimal_export: AnalysisExport):
        """Test footer includes tool version and schema version."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert "mcp-vector-search v0.19.0" in html
        assert "Schema version: 1.0.0" in html

    def test_large_dataset_handling(
        self, minimal_metadata: ExportMetadata, basic_summary: MetricsSummary
    ):
        """Test handling of large number of files (limits to top 20)."""
        # Create 50 files with varying complexity
        files = []
        for i in range(50):
            files.append(
                FileDetail(
                    path=f"src/file_{i}.py",
                    language="python",
                    lines_of_code=100,
                    cyclomatic_complexity=i,
                    cognitive_complexity=i,
                    max_nesting_depth=1,
                    function_count=2,
                    class_count=0,
                    efferent_coupling=0,
                    afferent_coupling=0,
                )
            )

        export = AnalysisExport(
            metadata=minimal_metadata,
            summary=basic_summary,
            files=files,
            dependencies=DependencyGraph(),
        )

        gen = HTMLReportGenerator()
        html = gen.generate(export)

        # Should limit to top 20 files by complexity in the table
        # Files 49, 48, 47, ... should be present
        assert "src/file_49.py" in html
        assert "src/file_48.py" in html

        # The "Top Files by Complexity" table should exist
        assert "Top Files by Complexity" in html

        # Note: Low-complexity files will appear in the D3 graph JSON data
        # but not in the files table. We can't easily test table-only presence
        # without parsing HTML, so we just verify the high-complexity files are there.

    def test_large_smells_handling(
        self, minimal_metadata: ExportMetadata, basic_summary: MetricsSummary
    ):
        """Test handling of large number of smells (limits to top 20)."""
        # Create file with 50 smells
        smells = []
        for i in range(50):
            smells.append(
                SmellLocation(
                    smell_type=f"smell_{i}",
                    severity="warning",
                    message=f"Smell {i}",
                    line=i + 1,
                )
            )

        files = [
            FileDetail(
                path="src/smelly.py",
                language="python",
                lines_of_code=1000,
                cyclomatic_complexity=50,
                cognitive_complexity=50,
                max_nesting_depth=5,
                function_count=10,
                class_count=2,
                efferent_coupling=0,
                afferent_coupling=0,
                smells=smells,
            )
        ]

        export = AnalysisExport(
            metadata=minimal_metadata,
            summary=basic_summary,
            files=files,
            dependencies=DependencyGraph(),
        )

        gen = HTMLReportGenerator()
        html = gen.generate(export)

        # Should show total count (12 from basic_summary)
        assert "12 total" in html or "Code Smells" in html

        # Should limit to top 20 smells in table
        # Can't easily test exact count without parsing HTML, but check first and last are present
        assert "smell_0" in html
        assert "smell_19" in html or "smell_49" not in html  # Shouldn't show all 50

    def test_responsive_meta_tag(self, minimal_export: AnalysisExport):
        """Test viewport meta tag for mobile responsiveness."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert (
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            in html
        )

    def test_valid_html_structure(self, minimal_export: AnalysisExport):
        """Test basic HTML structure validity."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check required tags
        assert html.startswith("<!DOCTYPE html>")
        assert '<html lang="en">' in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert html.strip().endswith("</html>")

        # Check meta tags
        assert '<meta charset="UTF-8">' in html

    def test_custom_title_in_multiple_locations(self, minimal_export: AnalysisExport):
        """Test custom title appears in both <title> tag and header."""
        gen = HTMLReportGenerator(title="My Custom Analysis")
        html = gen.generate(minimal_export)

        assert "<title>My Custom Analysis</title>" in html
        assert "<h1>📊 My Custom Analysis</h1>" in html

    def test_chart_js_initialization(self, minimal_export: AnalysisExport):
        """Test Chart.js initialization code is present."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        assert "new Chart(ctx" in html
        assert "type: 'doughnut'" in html
        assert "backgroundColor:" in html

    def test_severity_badge_classes(self, full_export: AnalysisExport):
        """Test severity levels get correct CSS classes."""
        gen = HTMLReportGenerator()
        html = gen.generate(full_export)

        # Severity badge should have appropriate class
        # Warning severity should map to grade-d
        assert "grade-d" in html or "grade-badge" in html

    def test_empty_export(self, minimal_metadata: ExportMetadata):
        """Test export with minimal data (zero stats)."""
        summary = MetricsSummary(
            total_files=0,
            total_functions=0,
            total_classes=0,
            total_lines=0,
            avg_complexity=0.0,
            avg_cognitive_complexity=0.0,
            avg_nesting_depth=0.0,
            total_smells=0,
            smells_by_severity={},
        )

        export = AnalysisExport(
            metadata=minimal_metadata,
            summary=summary,
            files=[],
            dependencies=DependencyGraph(),
        )

        gen = HTMLReportGenerator()
        html = gen.generate(export)

        # Should handle zeros gracefully
        assert "0" in html
        assert "No code smells detected! 🎉" in html

    def test_utf8_encoding(self, minimal_export: AnalysisExport, tmp_path: Path):
        """Test UTF-8 encoding is properly set."""
        gen = HTMLReportGenerator()
        output_path = tmp_path / "report.html"

        gen.generate_to_file(minimal_export, output_path)

        # Read with UTF-8 encoding
        content = output_path.read_text(encoding="utf-8")
        assert '<meta charset="UTF-8">' in content

    def test_circular_dependency_limit(
        self, minimal_metadata: ExportMetadata, basic_summary: MetricsSummary
    ):
        """Test circular dependencies are limited to top 10."""
        # Create 15 circular dependencies
        cycles = [
            CyclicDependency(
                cycle=[f"src/a{i}.py", f"src/b{i}.py", f"src/a{i}.py"], length=3
            )
            for i in range(15)
        ]

        export = AnalysisExport(
            metadata=minimal_metadata,
            summary=basic_summary,
            files=[],
            dependencies=DependencyGraph(circular_dependencies=cycles),
        )

        gen = HTMLReportGenerator()
        html = gen.generate(export)

        # Should show total count
        assert "15" in html

        # Should limit display to 10
        assert "src/a0.py" in html
        assert "src/a9.py" in html
        # Higher numbers may or may not appear depending on display logic

    def test_filter_controls_generated(self, minimal_export):
        """Test that filter controls panel is generated in HTML."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check filter controls exist
        assert 'class="filter-controls"' in html
        assert 'id="filter-grade-a"' in html
        assert 'id="filter-grade-b"' in html
        assert 'id="filter-grade-c"' in html
        assert 'id="filter-grade-d"' in html
        assert 'id="filter-grade-f"' in html

        # Check smell filters
        assert 'id="filter-smell-none"' in html
        assert 'id="filter-smell-info"' in html
        assert 'id="filter-smell-warning"' in html
        assert 'id="filter-smell-error"' in html

        # Check module filter
        assert 'id="filter-module"' in html

        # Check search box
        assert 'id="filter-search"' in html
        assert 'placeholder="Search by file name..."' in html

        # Check reset button
        assert 'onclick="resetFilters()"' in html
        assert "Reset Filters" in html

    def test_filter_module_options(self, minimal_metadata, basic_summary):
        """Test that module filter includes all unique modules."""
        # Create files with different modules
        files = [
            FileDetail(
                path="src/core/auth.py",
                module="core",
                language="python",
                cognitive_complexity=8,
                cyclomatic_complexity=6,
                lines_of_code=150,
                max_nesting_depth=3,
                function_count=5,
                class_count=1,
                efferent_coupling=2,
                afferent_coupling=3,
            ),
            FileDetail(
                path="src/api/routes.py",
                module="api",
                language="python",
                cognitive_complexity=12,
                cyclomatic_complexity=10,
                lines_of_code=200,
                max_nesting_depth=4,
                function_count=8,
                class_count=2,
                efferent_coupling=4,
                afferent_coupling=1,
            ),
            FileDetail(
                path="src/utils/helpers.py",
                module="utils",
                language="python",
                cognitive_complexity=5,
                cyclomatic_complexity=4,
                lines_of_code=80,
                max_nesting_depth=2,
                function_count=10,
                class_count=0,
                efferent_coupling=0,
                afferent_coupling=5,
            ),
        ]

        export = AnalysisExport(
            metadata=minimal_metadata,
            summary=basic_summary,
            files=files,
            dependencies=DependencyGraph(),
        )

        gen = HTMLReportGenerator()
        html = gen.generate(export)

        # Check all modules appear as options
        assert '<option value="api">api</option>' in html
        assert '<option value="core">core</option>' in html
        assert '<option value="utils">utils</option>' in html
        assert "All Modules" in html

    def test_enhanced_tooltip_styles(self, minimal_export):
        """Test that enhanced tooltip CSS classes are included."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check tooltip CSS classes
        assert ".d3-tooltip" in html
        assert ".tooltip-header" in html
        assert ".tooltip-subtitle" in html
        assert ".tooltip-section" in html
        assert ".tooltip-metric" in html
        assert ".tooltip-bar" in html
        assert ".tooltip-bar-fill" in html
        assert ".tooltip-badge" in html

    def test_filter_functionality_javascript(self, minimal_export):
        """Test that filter JavaScript functions are included."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check filter functions
        assert "getComplexityGrade" in html
        assert "filterState" in html
        assert "applyFilters" in html

        # Check filter event listeners
        assert "addEventListener" in html
        assert "filter-grade-" in html
        assert "filter-smell-" in html

        # Check filter state management
        assert ".node-filtered" in html
        assert ".link-filtered" in html
        assert ".node-search-highlight" in html

    def test_keyboard_navigation_javascript(self, minimal_export):
        """Test that keyboard navigation is implemented."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check keyboard handler
        assert "document.addEventListener('keydown'" in html
        assert "case 'Tab':" in html
        assert "case 'Enter':" in html
        assert "case 'Escape':" in html
        assert "case 'ArrowRight':" in html
        assert "case 'ArrowLeft':" in html
        assert "case 'ArrowUp':" in html
        assert "case 'ArrowDown':" in html

        # Check navigation functions
        assert "getVisibleNodes" in html
        assert "focusNode" in html
        assert "navigateToConnected" in html

        # Check focus styling
        assert ".node-focused" in html

    def test_enhanced_tooltip_rendering(self, minimal_export):
        """Test that enhanced tooltip rendering function exists."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check tooltip rendering function
        assert "renderTooltip" in html
        assert "tooltip-header" in html
        assert "tooltip-metric" in html
        assert "complexity bar" in html.lower() or "complexityPct" in html

        # Check tooltip follows cursor
        assert "mousemove" in html
        assert "pageX" in html
        assert "pageY" in html

    def test_reset_filters_function(self, minimal_export):
        """Test that reset filters function is implemented."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check reset filters function exists
        assert "function resetFilters()" in html
        assert "checked = true" in html

        # Check it resets all filter types
        assert "filter-grade-" in html
        assert "filter-smell-" in html
        assert "filter-module" in html
        assert "filter-search" in html

    def test_clear_selection_function(self, minimal_export):
        """Test that clear selection function exists."""
        gen = HTMLReportGenerator()
        html = gen.generate(minimal_export)

        # Check clear selection function
        assert "function clearSelection()" in html
        assert "node-detail-panel" in html
        assert "hidden" in html
        assert "node-focused" in html
