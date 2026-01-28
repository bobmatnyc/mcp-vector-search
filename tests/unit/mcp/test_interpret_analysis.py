"""Tests for MCP interpret_analysis tool."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from mcp_vector_search.analysis.visualizer.schemas import (
    AnalysisExport,
    DependencyGraph,
    ExportMetadata,
    MetricsSummary,
)
from mcp_vector_search.mcp.server import MCPVectorSearchServer


@pytest.fixture
def mcp_server(tmp_path: Path) -> MCPVectorSearchServer:
    """Create MCP server for testing."""
    return MCPVectorSearchServer(project_root=tmp_path, enable_file_watching=False)


@pytest.fixture
def sample_analysis_json(tmp_path: Path) -> str:
    """Create sample analysis JSON with LLM context."""
    from mcp_vector_search.analysis.interpretation import LLMContextExport

    # Create minimal analysis
    metadata = ExportMetadata(
        version="1.0.0",
        generated_at=datetime.now(),
        tool_version="0.19.0",
        project_root=str(tmp_path),
    )

    summary = MetricsSummary(
        total_files=5,
        total_functions=25,
        total_classes=8,
        total_lines=1000,
        avg_complexity=8.5,
        avg_cognitive_complexity=10.2,
        avg_nesting_depth=2.5,
        total_smells=10,
        smells_by_severity={"error": 2, "warning": 6, "info": 2},
    )

    analysis = AnalysisExport(
        metadata=metadata,
        summary=summary,
        files=[],
        dependencies=DependencyGraph(),
    )

    export = LLMContextExport(
        analysis=analysis,
        threshold_comparisons={
            "complexity": [
                {
                    "metric_name": "avg_complexity",
                    "value": 8.5,
                    "threshold": 10.0,
                    "comparison": "below",
                    "percentage_of_threshold": 85.0,
                    "severity": "info",
                    "interpretation": "avg_complexity is within acceptable range",
                }
            ]
        },
        remediation_summary={
            "total_smells": 10,
            "smells_by_severity": {"error": 2, "warning": 6, "info": 2},
            "priority_counts": {"critical": 2, "high": 6, "medium": 2, "low": 0},
            "estimated_remediation_hours": 3.5,
            "recommended_focus": ["Address error-level smells"],
        },
        code_quality_grade="B",
        interpretation_hints=["Small project - focus on good patterns"],
    )

    return export.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestInterpretAnalysisTool:
    """Tests for interpret_analysis MCP tool."""

    async def test_tool_registration(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test that interpret_analysis tool is registered."""
        tools = mcp_server.get_tools()
        tool_names = [t.name for t in tools]

        assert "interpret_analysis" in tool_names

    async def test_tool_schema(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test interpret_analysis tool schema."""
        tools = mcp_server.get_tools()
        interpret_tool = next(t for t in tools if t.name == "interpret_analysis")

        # Verify required parameters
        assert "analysis_json" in interpret_tool.inputSchema["properties"]
        assert "focus" in interpret_tool.inputSchema["properties"]
        assert "verbosity" in interpret_tool.inputSchema["properties"]

        # Verify enums
        assert interpret_tool.inputSchema["properties"]["focus"]["enum"] == [
            "summary",
            "recommendations",
            "priorities",
        ]
        assert interpret_tool.inputSchema["properties"]["verbosity"]["enum"] == [
            "brief",
            "normal",
            "detailed",
        ]

    async def test_interpret_summary(
        self, mcp_server: MCPVectorSearchServer, sample_analysis_json: str
    ) -> None:
        """Test summary interpretation."""
        from types import SimpleNamespace

        # Create mock request
        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {
            "analysis_json": sample_analysis_json,
            "focus": "summary",
            "verbosity": "normal",
        }

        result = await mcp_server.call_tool(request)

        assert not result.isError
        assert len(result.content) > 0

        # Verify content structure
        text = result.content[0].text
        assert "Code Quality Assessment" in text
        assert "Grade B" in text
        assert "Project Size" in text

    async def test_interpret_recommendations(
        self, mcp_server: MCPVectorSearchServer, sample_analysis_json: str
    ) -> None:
        """Test recommendations interpretation."""
        from types import SimpleNamespace

        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {
            "analysis_json": sample_analysis_json,
            "focus": "recommendations",
            "verbosity": "normal",
        }

        result = await mcp_server.call_tool(request)

        assert not result.isError

        text = result.content[0].text
        assert "Recommended Actions" in text
        assert "Estimated Effort" in text
        assert "3.5 hours" in text

    async def test_interpret_priorities(
        self, mcp_server: MCPVectorSearchServer, sample_analysis_json: str
    ) -> None:
        """Test priorities interpretation."""
        from types import SimpleNamespace

        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {
            "analysis_json": sample_analysis_json,
            "focus": "priorities",
            "verbosity": "normal",
        }

        result = await mcp_server.call_tool(request)

        assert not result.isError

        text = result.content[0].text
        assert "Remediation Priorities" in text
        assert "By Priority Level" in text

    async def test_verbosity_levels(
        self, mcp_server: MCPVectorSearchServer, sample_analysis_json: str
    ) -> None:
        """Test different verbosity levels."""
        from types import SimpleNamespace

        # Test brief
        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {
            "analysis_json": sample_analysis_json,
            "focus": "summary",
            "verbosity": "brief",
        }

        result_brief = await mcp_server.call_tool(request)
        assert not result_brief.isError

        # Test detailed
        request.params.arguments["verbosity"] = "detailed"
        result_detailed = await mcp_server.call_tool(request)
        assert not result_detailed.isError

        # Detailed should be longer
        assert len(result_detailed.content[0].text) >= len(result_brief.content[0].text)

    async def test_missing_analysis_json(
        self, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test error handling when analysis_json is missing."""
        from types import SimpleNamespace

        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {"focus": "summary"}

        result = await mcp_server.call_tool(request)

        assert result.isError
        assert "required" in result.content[0].text.lower()

    async def test_invalid_json(self, mcp_server: MCPVectorSearchServer) -> None:
        """Test error handling for invalid JSON."""
        from types import SimpleNamespace

        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {
            "analysis_json": "{ invalid json }",
            "focus": "summary",
        }

        result = await mcp_server.call_tool(request)

        assert result.isError
        assert (
            "Invalid JSON" in result.content[0].text
            or "failed" in result.content[0].text.lower()
        )

    async def test_default_parameters(
        self, mcp_server: MCPVectorSearchServer, sample_analysis_json: str
    ) -> None:
        """Test that default parameters work correctly."""
        from types import SimpleNamespace

        # Only provide required parameter
        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {"analysis_json": sample_analysis_json}

        result = await mcp_server.call_tool(request)

        # Should default to focus="summary", verbosity="normal"
        assert not result.isError
        assert "Code Quality Assessment" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - MCP server handlers extracted to separate modules"
)
class TestIntegrationWorkflow:
    """Integration tests for complete analyze + interpret workflow."""

    async def test_cli_to_mcp_workflow(
        self, tmp_path: Path, mcp_server: MCPVectorSearchServer
    ) -> None:
        """Test workflow from CLI analyze to MCP interpret."""
        # This would require creating actual Python files and running analysis
        # For now, we test the schema compatibility

        from mcp_vector_search.analysis.interpretation import EnhancedJSONExporter
        from mcp_vector_search.analysis.metrics import (
            ChunkMetrics,
            FileMetrics,
            ProjectMetrics,
        )

        # Create sample project metrics
        metrics = ProjectMetrics(project_root=str(tmp_path))
        file1 = FileMetrics(file_path=str(tmp_path / "test.py"))
        file1.total_lines = 100
        file1.function_count = 5

        chunk = ChunkMetrics(
            cognitive_complexity=8,
            cyclomatic_complexity=6,
            max_nesting_depth=3,
            parameter_count=4,
            lines_of_code=20,
        )
        file1.chunks = [chunk]
        file1.compute_aggregates()

        metrics.files[str(tmp_path / "test.py")] = file1
        metrics.compute_aggregates()

        # Export with context (simulating CLI --include-context)
        exporter = EnhancedJSONExporter(project_root=tmp_path)
        export = exporter.export_with_context(metrics)

        # Convert to JSON string (as CLI would output)
        analysis_json = export.model_dump_json()

        # Now interpret via MCP (simulating LLM consuming the JSON)
        from types import SimpleNamespace

        request = SimpleNamespace()
        request.params = SimpleNamespace()
        request.params.name = "interpret_analysis"
        request.params.arguments = {
            "analysis_json": analysis_json,
            "focus": "summary",
            "verbosity": "normal",
        }

        result = await mcp_server.call_tool(request)

        # Verify end-to-end works
        assert not result.isError
        assert "Code Quality Assessment" in result.content[0].text
