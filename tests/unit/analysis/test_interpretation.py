"""Tests for LLM-friendly interpretation of code analysis results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mcp_vector_search.analysis.interpretation import (
    AnalysisInterpreter,
    EnhancedJSONExporter,
    ThresholdComparison,
    ThresholdContext,
)
from mcp_vector_search.analysis.metrics import ChunkMetrics, FileMetrics, ProjectMetrics
from mcp_vector_search.analysis.visualizer.schemas import (
    AnalysisExport,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    MetricsSummary,
    SmellLocation,
)


class TestThresholdContext:
    """Tests for threshold comparison context."""

    def test_well_below_threshold(self) -> None:
        """Test threshold context when value is well below threshold."""
        context = ThresholdContext(
            metric_name="complexity",
            value=5.0,
            threshold=20.0,
            comparison=ThresholdComparison.WELL_BELOW,
            percentage_of_threshold=25.0,
            severity="info",
        )

        assert context.comparison == ThresholdComparison.WELL_BELOW
        assert "below threshold (healthy)" in context.get_interpretation()

    def test_above_threshold(self) -> None:
        """Test threshold context when value exceeds threshold."""
        context = ThresholdContext(
            metric_name="complexity",
            value=25.0,
            threshold=20.0,
            comparison=ThresholdComparison.ABOVE,
            percentage_of_threshold=125.0,
            severity="warning",
        )

        assert context.comparison == ThresholdComparison.ABOVE
        assert "exceeds threshold" in context.get_interpretation()
        assert "needs attention" in context.get_interpretation()

    def test_to_dict(self) -> None:
        """Test conversion to dictionary for JSON serialization."""
        context = ThresholdContext(
            metric_name="complexity",
            value=15.0,
            threshold=10.0,
            comparison=ThresholdComparison.ABOVE,
            percentage_of_threshold=150.0,
            severity="warning",
        )

        result = context.to_dict()
        assert result["metric_name"] == "complexity"
        assert result["value"] == 15.0
        assert result["threshold"] == 10.0
        assert result["comparison"] == "above"
        assert result["percentage_of_threshold"] == 150.0
        assert result["severity"] == "warning"
        assert "interpretation" in result


class TestEnhancedJSONExporter:
    """Tests for enhanced JSON exporter with LLM context."""

    @pytest.fixture
    def sample_project_metrics(self, tmp_path: Path) -> ProjectMetrics:
        """Create sample project metrics for testing."""
        metrics = ProjectMetrics(project_root=str(tmp_path))

        # Add sample file metrics
        file1 = FileMetrics(file_path=str(tmp_path / "file1.py"))
        file1.total_lines = 100
        file1.code_lines = 80
        file1.function_count = 5
        file1.class_count = 1

        # Add sample chunks
        chunk1 = ChunkMetrics(
            cognitive_complexity=10,
            cyclomatic_complexity=8,
            max_nesting_depth=3,
            parameter_count=4,
            lines_of_code=20,
        )
        chunk2 = ChunkMetrics(
            cognitive_complexity=25,  # High complexity
            cyclomatic_complexity=18,
            max_nesting_depth=5,  # Deep nesting
            parameter_count=7,  # Many parameters
            lines_of_code=60,  # Long method
        )
        file1.chunks = [chunk1, chunk2]
        file1.compute_aggregates()

        metrics.files[str(tmp_path / "file1.py")] = file1
        metrics.compute_aggregates()

        return metrics

    def test_export_with_context(
        self, sample_project_metrics: ProjectMetrics, tmp_path: Path
    ) -> None:
        """Test enhanced export with LLM context."""
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(sample_project_metrics)

        # Verify enhanced export structure
        assert export.analysis is not None
        assert export.threshold_comparisons is not None
        assert export.remediation_summary is not None
        assert export.code_quality_grade in ["A", "B", "C", "D", "F"]
        assert isinstance(export.interpretation_hints, list)

    def test_threshold_comparisons(
        self, sample_project_metrics: ProjectMetrics, tmp_path: Path
    ) -> None:
        """Test threshold comparison generation."""
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(sample_project_metrics)

        # Should have complexity and size comparisons
        assert "complexity" in export.threshold_comparisons
        assert "size" in export.threshold_comparisons

        # Verify structure
        for comparison in export.threshold_comparisons["complexity"]:
            assert "metric_name" in comparison
            assert "value" in comparison
            assert "threshold" in comparison
            assert "comparison" in comparison
            assert "interpretation" in comparison

    def test_remediation_summary(
        self, sample_project_metrics: ProjectMetrics, tmp_path: Path
    ) -> None:
        """Test remediation summary generation."""
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(sample_project_metrics)

        summary = export.remediation_summary
        assert "total_smells" in summary
        assert "smells_by_severity" in summary
        assert "priority_counts" in summary
        assert "estimated_remediation_hours" in summary
        assert "recommended_focus" in summary
        assert isinstance(summary["recommended_focus"], list)

    def test_quality_grade_calculation(
        self, sample_project_metrics: ProjectMetrics, tmp_path: Path
    ) -> None:
        """Test code quality grade calculation."""
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        grade = exporter._calculate_quality_grade(sample_project_metrics)

        assert grade in ["A", "B", "C", "D", "F"]

    def test_interpretation_hints(
        self, sample_project_metrics: ProjectMetrics, tmp_path: Path
    ) -> None:
        """Test interpretation hints generation."""
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(sample_project_metrics)

        hints = export.interpretation_hints
        assert isinstance(hints, list)
        # Should provide context about project size
        assert any("project" in hint.lower() for hint in hints)


class TestAnalysisInterpreter:
    """Tests for natural language analysis interpreter."""

    @pytest.fixture
    def sample_llm_export(self, tmp_path: Path) -> LLMContextExport:
        """Create sample LLM context export for testing."""
        from mcp_vector_search.analysis.interpretation import LLMContextExport

        # Create minimal analysis export
        metadata = ExportMetadata(
            version="1.0.0",
            generated_at=datetime.now(),
            tool_version="0.19.0",
            project_root=str(tmp_path),
        )

        summary = MetricsSummary(
            total_files=10,
            total_functions=50,
            total_classes=15,
            total_lines=2000,
            avg_complexity=12.5,
            avg_cognitive_complexity=15.2,
            avg_nesting_depth=3.1,
            total_smells=25,
            smells_by_severity={"error": 5, "warning": 15, "info": 5},
            avg_instability=0.6,
            circular_dependencies=2,
        )

        # Create file with smells
        file_detail = FileDetail(
            path="src/example.py",
            language="python",
            lines_of_code=200,
            cyclomatic_complexity=25,
            cognitive_complexity=30,
            max_nesting_depth=5,
            function_count=10,
            class_count=2,
            efferent_coupling=5,
            afferent_coupling=3,
            instability=0.625,
            smells=[
                SmellLocation(
                    smell_type="Long Method",
                    severity="warning",
                    message="Method too long",
                    line=50,
                ),
                SmellLocation(
                    smell_type="Deep Nesting",
                    severity="warning",
                    message="Excessive nesting",
                    line=75,
                ),
            ],
        )

        analysis = AnalysisExport(
            metadata=metadata,
            summary=summary,
            files=[file_detail],
            dependencies=DependencyGraph(),
        )

        return LLMContextExport(
            analysis=analysis,
            threshold_comparisons={
                "complexity": [
                    {
                        "metric_name": "avg_complexity",
                        "value": 12.5,
                        "threshold": 10.0,
                        "comparison": "above",
                        "percentage_of_threshold": 125.0,
                        "severity": "warning",
                        "interpretation": "avg_complexity exceeds threshold by 25% (needs attention)",
                    }
                ]
            },
            remediation_summary={
                "total_smells": 25,
                "smells_by_severity": {"error": 5, "warning": 15, "info": 5},
                "priority_counts": {"critical": 5, "high": 15, "medium": 5, "low": 0},
                "estimated_remediation_hours": 12.5,
                "recommended_focus": [
                    "Reduce overall complexity",
                    "Address circular dependencies",
                ],
            },
            code_quality_grade="C",
            interpretation_hints=["Medium project - maintain modularity"],
        )

    def test_interpret_summary(self, sample_llm_export: LLMContextExport) -> None:
        """Test summary interpretation generation."""
        interpreter = AnalysisInterpreter()
        result = interpreter.interpret(
            sample_llm_export, focus="summary", verbosity="normal"
        )

        # Verify output contains expected sections
        assert "Code Quality Assessment" in result
        assert "Grade C" in result
        assert "Project Size" in result
        assert "Average Complexity" in result
        assert "Code Smells" in result

    def test_interpret_recommendations(
        self, sample_llm_export: LLMContextExport
    ) -> None:
        """Test recommendations interpretation generation."""
        interpreter = AnalysisInterpreter()
        result = interpreter.interpret(
            sample_llm_export, focus="recommendations", verbosity="normal"
        )

        # Verify recommendations structure
        assert "Recommended Actions" in result
        assert "Estimated Effort" in result
        assert "Priority Focus Areas" in result
        assert "12.5 hours" in result

    def test_interpret_priorities(self, sample_llm_export: LLMContextExport) -> None:
        """Test priorities interpretation generation."""
        interpreter = AnalysisInterpreter()
        result = interpreter.interpret(
            sample_llm_export, focus="priorities", verbosity="normal"
        )

        # Verify priorities structure
        assert "Remediation Priorities" in result
        assert "By Priority Level" in result
        assert "Critical:" in result
        assert "High:" in result
        assert "Medium:" in result

    def test_interpret_verbosity_levels(
        self, sample_llm_export: LLMContextExport
    ) -> None:
        """Test different verbosity levels."""
        interpreter = AnalysisInterpreter()

        # Brief should be shorter
        _brief = interpreter.interpret(
            sample_llm_export, focus="summary", verbosity="brief"
        )
        normal = interpreter.interpret(
            sample_llm_export, focus="summary", verbosity="normal"
        )
        detailed = interpreter.interpret(
            sample_llm_export, focus="summary", verbosity="detailed"
        )

        # Detailed should include threshold comparisons
        assert "Threshold Comparisons" in detailed
        assert len(detailed) >= len(normal)

    def test_invalid_focus(self, sample_llm_export: LLMContextExport) -> None:
        """Test handling of invalid focus parameter."""
        interpreter = AnalysisInterpreter()
        # Should default to summary
        result = interpreter.interpret(
            sample_llm_export, focus="invalid", verbosity="normal"
        )

        assert "Code Quality Assessment" in result


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_end_to_end_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from metrics to interpretation."""
        # Create sample metrics
        metrics = ProjectMetrics(project_root=str(tmp_path))
        file1 = FileMetrics(file_path=str(tmp_path / "test.py"))
        file1.total_lines = 50
        file1.function_count = 3

        chunk = ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=4,
            max_nesting_depth=2,
            parameter_count=3,
            lines_of_code=15,
        )
        file1.chunks = [chunk]
        file1.compute_aggregates()

        metrics.files[str(tmp_path / "test.py")] = file1
        metrics.compute_aggregates()

        # Export with context
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(metrics)

        # Interpret
        interpreter = AnalysisInterpreter()
        summary = interpreter.interpret(export, focus="summary")
        recommendations = interpreter.interpret(export, focus="recommendations")
        priorities = interpreter.interpret(export, focus="priorities")

        # Verify all interpretations generated
        assert len(summary) > 0
        assert len(recommendations) > 0
        assert len(priorities) > 0

        # Verify interpretations are different
        assert summary != recommendations
        assert summary != priorities

    def test_json_serialization(self, tmp_path: Path) -> None:
        """Test that enhanced export can be JSON serialized."""
        import json

        metrics = ProjectMetrics(project_root=str(tmp_path))
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(metrics)

        # Should be serializable
        json_str = export.model_dump_json()
        assert len(json_str) > 0

        # Should be deserializable
        data = json.loads(json_str)
        assert "analysis" in data
        assert "threshold_comparisons" in data
        assert "remediation_summary" in data
