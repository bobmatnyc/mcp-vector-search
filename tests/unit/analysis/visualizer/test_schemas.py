"""Tests for JSON export schema models.

Tests cover:
- Model instantiation with valid data
- Validation of invalid data
- JSON serialization/deserialization roundtrip
- Schema generation
- Optional field handling
- Field validators and constraints
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mcp_vector_search.analysis.visualizer.schemas import (
    AnalysisExport,
    ClassMetrics,
    CyclicDependency,
    DependencyEdge,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    FunctionMetrics,
    MetricsSummary,
    MetricTrend,
    SmellLocation,
    TrendData,
    TrendDataPoint,
    generate_json_schema,
)


class TestExportMetadata:
    """Tests for ExportMetadata model."""

    def test_valid_metadata(self):
        """Test creation with all required fields."""
        metadata = ExportMetadata(
            version="1.0.0",
            generated_at=datetime.now(UTC),
            tool_version="0.19.0",
            project_root="/path/to/project",
            git_commit="abc123def456",
            git_branch="main",
        )
        assert metadata.version == "1.0.0"
        assert metadata.tool_version == "0.19.0"
        assert metadata.project_root == "/path/to/project"
        assert metadata.git_commit == "abc123def456"
        assert metadata.git_branch == "main"

    def test_optional_git_fields(self):
        """Test that git fields are optional."""
        metadata = ExportMetadata(
            generated_at=datetime.now(UTC),
            tool_version="0.19.0",
            project_root="/path/to/project",
        )
        assert metadata.git_commit is None
        assert metadata.git_branch is None
        assert metadata.version == "1.0.0"  # Default value

    def test_json_serialization(self):
        """Test JSON serialization roundtrip."""
        metadata = ExportMetadata(
            generated_at=datetime.now(UTC),
            tool_version="0.19.0",
            project_root="/path/to/project",
        )
        json_str = metadata.model_dump_json()
        parsed = ExportMetadata.model_validate_json(json_str)
        assert parsed.tool_version == metadata.tool_version
        assert parsed.project_root == metadata.project_root


class TestMetricsSummary:
    """Tests for MetricsSummary model."""

    def test_valid_summary(self):
        """Test creation with all required fields."""
        summary = MetricsSummary(
            total_files=100,
            total_functions=500,
            total_classes=50,
            total_lines=10000,
            avg_complexity=5.5,
            avg_cognitive_complexity=8.2,
            avg_nesting_depth=2.3,
            total_smells=25,
            smells_by_severity={"error": 5, "warning": 15, "info": 5},
            avg_instability=0.5,
            circular_dependencies=2,
        )
        assert summary.total_files == 100
        assert summary.total_functions == 500
        assert summary.avg_complexity == 5.5
        assert summary.circular_dependencies == 2

    def test_optional_fields(self):
        """Test that optional metrics are handled correctly."""
        summary = MetricsSummary(
            total_files=10,
            total_functions=50,
            total_classes=5,
            total_lines=1000,
            avg_complexity=3.0,
            avg_cognitive_complexity=4.0,
            avg_nesting_depth=1.5,
            total_smells=5,
        )
        assert summary.avg_instability is None
        assert summary.avg_halstead_volume is None
        assert summary.estimated_debt_minutes is None

    def test_negative_values_rejected(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MetricsSummary(
                total_files=-1,  # Invalid
                total_functions=50,
                total_classes=5,
                total_lines=1000,
                avg_complexity=3.0,
                avg_cognitive_complexity=4.0,
                avg_nesting_depth=1.5,
                total_smells=5,
            )
        assert "greater_than_equal" in str(exc_info.value)

    def test_instability_range_validation(self):
        """Test that instability is constrained to 0-1."""
        # Valid: within range
        summary = MetricsSummary(
            total_files=10,
            total_functions=50,
            total_classes=5,
            total_lines=1000,
            avg_complexity=3.0,
            avg_cognitive_complexity=4.0,
            avg_nesting_depth=1.5,
            total_smells=5,
            avg_instability=0.8,
        )
        assert summary.avg_instability == 0.8

        # Invalid: above 1.0
        with pytest.raises(ValidationError):
            MetricsSummary(
                total_files=10,
                total_functions=50,
                total_classes=5,
                total_lines=1000,
                avg_complexity=3.0,
                avg_cognitive_complexity=4.0,
                avg_nesting_depth=1.5,
                total_smells=5,
                avg_instability=1.5,  # Invalid
            )


class TestFunctionMetrics:
    """Tests for FunctionMetrics model."""

    def test_valid_function(self):
        """Test creation with valid data."""
        func = FunctionMetrics(
            name="calculate_total",
            line_start=10,
            line_end=25,
            cyclomatic_complexity=5,
            cognitive_complexity=8,
            nesting_depth=2,
            parameter_count=3,
            lines_of_code=15,
        )
        assert func.name == "calculate_total"
        assert func.line_start == 10
        assert func.line_end == 25
        assert func.cyclomatic_complexity == 5

    def test_line_range_validation(self):
        """Test that line_end must be >= line_start."""
        # Valid: line_end >= line_start
        func = FunctionMetrics(
            name="test",
            line_start=10,
            line_end=10,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            nesting_depth=0,
            parameter_count=0,
            lines_of_code=1,
        )
        assert func.line_end == func.line_start

        # Invalid: line_end < line_start
        with pytest.raises(ValidationError) as exc_info:
            FunctionMetrics(
                name="test",
                line_start=25,
                line_end=10,  # Invalid
                cyclomatic_complexity=1,
                cognitive_complexity=1,
                nesting_depth=0,
                parameter_count=0,
                lines_of_code=1,
            )
        assert "line_end must be >= line_start" in str(exc_info.value)

    def test_optional_halstead_metrics(self):
        """Test that Halstead metrics are optional."""
        func = FunctionMetrics(
            name="test",
            line_start=1,
            line_end=10,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            nesting_depth=0,
            parameter_count=0,
            lines_of_code=10,
            halstead_volume=100.5,
            halstead_difficulty=5.2,
        )
        assert func.halstead_volume == 100.5
        assert func.halstead_difficulty == 5.2
        assert func.halstead_effort is None


class TestClassMetrics:
    """Tests for ClassMetrics model."""

    def test_valid_class(self):
        """Test creation with methods."""
        cls = ClassMetrics(
            name="UserService",
            line_start=10,
            line_end=100,
            method_count=5,
            lcom4=2,
            methods=[
                FunctionMetrics(
                    name="get_user",
                    line_start=15,
                    line_end=20,
                    cyclomatic_complexity=1,
                    cognitive_complexity=1,
                    nesting_depth=0,
                    parameter_count=1,
                    lines_of_code=5,
                )
            ],
        )
        assert cls.name == "UserService"
        assert cls.method_count == 5
        assert cls.lcom4 == 2
        assert len(cls.methods) == 1

    def test_line_range_validation(self):
        """Test that line_end must be >= line_start."""
        with pytest.raises(ValidationError) as exc_info:
            ClassMetrics(
                name="Test",
                line_start=100,
                line_end=10,  # Invalid
                method_count=0,
            )
        assert "line_end must be >= line_start" in str(exc_info.value)


class TestSmellLocation:
    """Tests for SmellLocation model."""

    def test_valid_smell(self):
        """Test creation with all fields."""
        smell = SmellLocation(
            smell_type="long_method",
            severity="warning",
            message="Method exceeds 50 lines",
            line=100,
            column=4,
            end_line=165,
            function_name="process_data",
            class_name="DataProcessor",
            remediation_minutes=30,
        )
        assert smell.smell_type == "long_method"
        assert smell.severity == "warning"
        assert smell.remediation_minutes == 30

    def test_severity_validation(self):
        """Test that severity must be error/warning/info."""
        # Valid severities
        for severity in ["error", "warning", "info"]:
            smell = SmellLocation(
                smell_type="test",
                severity=severity,
                message="test",
                line=1,
            )
            assert smell.severity == severity

        # Invalid severity
        with pytest.raises(ValidationError) as exc_info:
            SmellLocation(
                smell_type="test",
                severity="critical",  # Invalid
                message="test",
                line=1,
            )
        assert "severity must be one of" in str(exc_info.value)

    def test_optional_fields(self):
        """Test that many fields are optional."""
        smell = SmellLocation(
            smell_type="test",
            severity="info",
            message="test message",
            line=10,
        )
        assert smell.column is None
        assert smell.end_line is None
        assert smell.function_name is None
        assert smell.class_name is None
        assert smell.remediation_minutes is None


class TestFileDetail:
    """Tests for FileDetail model."""

    def test_valid_file(self):
        """Test creation with comprehensive data."""
        file = FileDetail(
            path="src/main.py",
            language="python",
            lines_of_code=150,
            cyclomatic_complexity=25,
            cognitive_complexity=30,
            max_nesting_depth=3,
            function_count=10,
            class_count=2,
            efferent_coupling=5,
            afferent_coupling=3,
            instability=0.625,
            functions=[
                FunctionMetrics(
                    name="main",
                    line_start=1,
                    line_end=10,
                    cyclomatic_complexity=2,
                    cognitive_complexity=3,
                    nesting_depth=1,
                    parameter_count=0,
                    lines_of_code=10,
                )
            ],
            smells=[
                SmellLocation(
                    smell_type="complex_method",
                    severity="warning",
                    message="High complexity",
                    line=50,
                )
            ],
            imports=["os", "sys", "pathlib"],
        )
        assert file.path == "src/main.py"
        assert file.language == "python"
        assert file.instability == 0.625
        assert len(file.functions) == 1
        assert len(file.smells) == 1
        assert len(file.imports) == 3

    def test_empty_collections(self):
        """Test file with no functions, classes, or smells."""
        file = FileDetail(
            path="config.py",
            language="python",
            lines_of_code=10,
            cyclomatic_complexity=0,
            cognitive_complexity=0,
            max_nesting_depth=0,
            function_count=0,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
        )
        assert len(file.functions) == 0
        assert len(file.classes) == 0
        assert len(file.smells) == 0
        assert len(file.imports) == 0


class TestDependencyEdge:
    """Tests for DependencyEdge model."""

    def test_valid_edge(self):
        """Test creation with valid import types."""
        for import_type in ["import", "from_import", "dynamic"]:
            edge = DependencyEdge(
                source="src/main.py",
                target="src/utils.py",
                import_type=import_type,
            )
            assert edge.import_type == import_type

    def test_invalid_import_type(self):
        """Test that import_type is validated."""
        with pytest.raises(ValidationError) as exc_info:
            DependencyEdge(
                source="src/main.py",
                target="src/utils.py",
                import_type="require",  # Invalid
            )
        assert "import_type must be one of" in str(exc_info.value)


class TestCyclicDependency:
    """Tests for CyclicDependency model."""

    def test_valid_cycle(self):
        """Test creation with matching length."""
        cycle = CyclicDependency(
            cycle=["a.py", "b.py", "c.py", "a.py"],
            length=4,
        )
        assert cycle.length == 4
        assert len(cycle.cycle) == 4

    def test_length_validation(self):
        """Test that length must match cycle length."""
        with pytest.raises(ValidationError) as exc_info:
            CyclicDependency(
                cycle=["a.py", "b.py", "a.py"],
                length=5,  # Doesn't match
            )
        assert "does not match cycle length" in str(exc_info.value)

    def test_minimum_cycle_length(self):
        """Test that cycles must have at least 2 elements."""
        with pytest.raises(ValidationError):
            CyclicDependency(
                cycle=["a.py"],  # Too short
                length=1,
            )


class TestDependencyGraph:
    """Tests for DependencyGraph model."""

    def test_valid_graph(self):
        """Test creation with all components."""
        graph = DependencyGraph(
            edges=[
                DependencyEdge(
                    source="a.py",
                    target="b.py",
                    import_type="import",
                )
            ],
            circular_dependencies=[
                CyclicDependency(
                    cycle=["a.py", "b.py", "a.py"],
                    length=3,
                )
            ],
            most_depended_on=[("utils.py", 10), ("config.py", 5)],
            most_dependent=[("main.py", 8), ("app.py", 6)],
        )
        assert len(graph.edges) == 1
        assert len(graph.circular_dependencies) == 1
        assert len(graph.most_depended_on) == 2
        assert len(graph.most_dependent) == 2

    def test_empty_graph(self):
        """Test empty dependency graph."""
        graph = DependencyGraph()
        assert len(graph.edges) == 0
        assert len(graph.circular_dependencies) == 0
        assert len(graph.most_depended_on) == 0
        assert len(graph.most_dependent) == 0


class TestTrendDataPoint:
    """Tests for TrendDataPoint model."""

    def test_valid_datapoint(self):
        """Test creation with commit."""
        point = TrendDataPoint(
            timestamp=datetime.now(UTC),
            commit="abc123",
            value=5.5,
        )
        assert point.commit == "abc123"
        assert point.value == 5.5

    def test_optional_commit(self):
        """Test that commit is optional."""
        point = TrendDataPoint(
            timestamp=datetime.now(UTC),
            value=5.5,
        )
        assert point.commit is None


class TestMetricTrend:
    """Tests for MetricTrend model."""

    def test_valid_trend(self):
        """Test creation with history."""
        trend = MetricTrend(
            metric_name="avg_complexity",
            current_value=5.5,
            previous_value=6.0,
            change_percent=-8.33,
            trend_direction="improving",
            history=[
                TrendDataPoint(
                    timestamp=datetime.now(UTC),
                    value=5.5,
                )
            ],
        )
        assert trend.metric_name == "avg_complexity"
        assert trend.trend_direction == "improving"
        assert len(trend.history) == 1

    def test_trend_direction_validation(self):
        """Test that trend_direction is validated."""
        # Valid directions
        for direction in ["improving", "worsening", "stable"]:
            trend = MetricTrend(
                metric_name="test",
                current_value=5.0,
                trend_direction=direction,
            )
            assert trend.trend_direction == direction

        # Invalid direction
        with pytest.raises(ValidationError) as exc_info:
            MetricTrend(
                metric_name="test",
                current_value=5.0,
                trend_direction="unknown",  # Invalid
            )
        assert "trend_direction must be one of" in str(exc_info.value)


class TestTrendData:
    """Tests for TrendData model."""

    def test_valid_trend_data(self):
        """Test creation with baseline."""
        trend_data = TrendData(
            metrics=[
                MetricTrend(
                    metric_name="complexity",
                    current_value=5.0,
                    trend_direction="stable",
                )
            ],
            baseline_name="main",
            baseline_date=datetime.now(UTC),
        )
        assert len(trend_data.metrics) == 1
        assert trend_data.baseline_name == "main"

    def test_optional_baseline(self):
        """Test that baseline fields are optional."""
        trend_data = TrendData()
        assert len(trend_data.metrics) == 0
        assert trend_data.baseline_name is None
        assert trend_data.baseline_date is None


class TestAnalysisExport:
    """Tests for AnalysisExport (root schema)."""

    def test_complete_export(self):
        """Test creation of complete export."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(UTC),
                tool_version="0.19.0",
                project_root="/project",
                git_commit="abc123",
                git_branch="main",
            ),
            summary=MetricsSummary(
                total_files=50,
                total_functions=200,
                total_classes=30,
                total_lines=5000,
                avg_complexity=4.5,
                avg_cognitive_complexity=6.0,
                avg_nesting_depth=1.8,
                total_smells=15,
            ),
            files=[
                FileDetail(
                    path="main.py",
                    language="python",
                    lines_of_code=100,
                    cyclomatic_complexity=10,
                    cognitive_complexity=12,
                    max_nesting_depth=2,
                    function_count=5,
                    class_count=1,
                    efferent_coupling=3,
                    afferent_coupling=2,
                )
            ],
            dependencies=DependencyGraph(
                edges=[
                    DependencyEdge(
                        source="main.py",
                        target="utils.py",
                        import_type="import",
                    )
                ]
            ),
        )
        assert export.metadata.tool_version == "0.19.0"
        assert export.summary.total_files == 50
        assert len(export.files) == 1
        assert len(export.dependencies.edges) == 1
        assert export.trends is None

    def test_with_trends(self):
        """Test export with optional trend data."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(UTC),
                tool_version="0.19.0",
                project_root="/project",
            ),
            summary=MetricsSummary(
                total_files=10,
                total_functions=50,
                total_classes=5,
                total_lines=1000,
                avg_complexity=3.0,
                avg_cognitive_complexity=4.0,
                avg_nesting_depth=1.5,
                total_smells=5,
            ),
            files=[],
            dependencies=DependencyGraph(),
            trends=TrendData(
                metrics=[
                    MetricTrend(
                        metric_name="complexity",
                        current_value=3.0,
                        trend_direction="stable",
                    )
                ],
                baseline_name="v1.0.0",
            ),
        )
        assert export.trends is not None
        assert len(export.trends.metrics) == 1

    def test_json_roundtrip(self):
        """Test full JSON serialization/deserialization roundtrip."""
        original = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(UTC),
                tool_version="0.19.0",
                project_root="/project",
            ),
            summary=MetricsSummary(
                total_files=10,
                total_functions=50,
                total_classes=5,
                total_lines=1000,
                avg_complexity=3.0,
                avg_cognitive_complexity=4.0,
                avg_nesting_depth=1.5,
                total_smells=5,
            ),
            files=[],
            dependencies=DependencyGraph(),
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        restored = AnalysisExport.model_validate_json(json_str)

        # Verify key fields match
        assert restored.metadata.tool_version == original.metadata.tool_version
        assert restored.summary.total_files == original.summary.total_files
        assert len(restored.files) == len(original.files)

    def test_json_pretty_print(self):
        """Test JSON export with indentation."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(UTC),
                tool_version="0.19.0",
                project_root="/project",
            ),
            summary=MetricsSummary(
                total_files=1,
                total_functions=5,
                total_classes=1,
                total_lines=100,
                avg_complexity=2.0,
                avg_cognitive_complexity=3.0,
                avg_nesting_depth=1.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(),
        )

        json_str = export.model_dump_json(indent=2)
        assert json_str.startswith("{\n")
        assert "  " in json_str  # Has indentation

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "summary" in parsed


class TestSchemaGeneration:
    """Tests for JSON Schema generation."""

    def test_generate_schema(self):
        """Test that schema generation works."""
        schema = generate_json_schema()

        assert isinstance(schema, dict)
        assert "$defs" in schema or "definitions" in schema
        assert "properties" in schema
        assert "metadata" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "files" in schema["properties"]
        assert "dependencies" in schema["properties"]

    def test_schema_is_json_serializable(self):
        """Test that generated schema can be serialized to JSON."""
        schema = generate_json_schema()
        json_str = json.dumps(schema, indent=2)
        assert json_str.startswith("{")

        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert parsed == schema


class TestModelDumpMethods:
    """Tests for Pydantic v2 dump methods."""

    def test_model_dump(self):
        """Test model_dump() returns dict."""
        summary = MetricsSummary(
            total_files=10,
            total_functions=50,
            total_classes=5,
            total_lines=1000,
            avg_complexity=3.0,
            avg_cognitive_complexity=4.0,
            avg_nesting_depth=1.5,
            total_smells=5,
        )

        data = summary.model_dump()
        assert isinstance(data, dict)
        assert data["total_files"] == 10
        assert data["total_functions"] == 50

    def test_model_dump_json(self):
        """Test model_dump_json() returns JSON string."""
        summary = MetricsSummary(
            total_files=10,
            total_functions=50,
            total_classes=5,
            total_lines=1000,
            avg_complexity=3.0,
            avg_cognitive_complexity=4.0,
            avg_nesting_depth=1.5,
            total_smells=5,
        )

        json_str = summary.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["total_files"] == 10

    def test_exclude_none(self):
        """Test excluding None values from dump."""
        summary = MetricsSummary(
            total_files=10,
            total_functions=50,
            total_classes=5,
            total_lines=1000,
            avg_complexity=3.0,
            avg_cognitive_complexity=4.0,
            avg_nesting_depth=1.5,
            total_smells=5,
        )

        # With None values
        data_with_none = summary.model_dump()
        assert "avg_instability" in data_with_none
        assert data_with_none["avg_instability"] is None

        # Without None values
        data_without_none = summary.model_dump(exclude_none=True)
        assert "avg_instability" not in data_without_none
