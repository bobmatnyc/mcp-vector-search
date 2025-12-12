"""Tests for D3.js visualization data transformation.

This module tests the transformation of analysis data into D3-friendly format
for interactive dependency graph visualization.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mcp_vector_search.analysis.visualizer import (
    AnalysisExport,
    CyclicDependency,
    DependencyEdge,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    MetricsSummary,
    SmellLocation,
)
from mcp_vector_search.analysis.visualizer.d3_data import (
    D3Edge,
    D3Node,
    _calculate_worst_severity,
    _create_edges,
    _create_node,
    _extract_circular_paths,
    get_complexity_class,
    get_smell_class,
    transform_for_d3,
)


@pytest.fixture
def sample_file_detail() -> FileDetail:
    """Create a sample FileDetail for testing."""
    return FileDetail(
        path="src/main.py",
        language="python",
        lines_of_code=150,
        cyclomatic_complexity=25,
        cognitive_complexity=18,
        max_nesting_depth=4,
        function_count=5,
        class_count=2,
        efferent_coupling=3,
        afferent_coupling=2,
        instability=0.6,
        smells=[
            SmellLocation(
                smell_type="long_method",
                severity="warning",
                message="Method too long",
                line=50,
            ),
            SmellLocation(
                smell_type="deep_nesting",
                severity="error",
                message="Nesting too deep",
                line=75,
            ),
        ],
        imports=["utils.helpers", "models.user"],
    )


@pytest.fixture
def sample_export() -> AnalysisExport:
    """Create a sample AnalysisExport for testing."""
    metadata = ExportMetadata(
        version="1.0.0",
        generated_at=datetime.now(),
        tool_version="0.19.0",
        project_root="/path/to/project",
    )

    summary = MetricsSummary(
        total_files=3,
        total_functions=15,
        total_classes=5,
        total_lines=450,
        avg_complexity=15.0,
        avg_cognitive_complexity=12.0,
        avg_nesting_depth=3.0,
        total_smells=8,
        smells_by_severity={"error": 2, "warning": 5, "info": 1},
        circular_dependencies=1,
    )

    files = [
        FileDetail(
            path="src/main.py",
            language="python",
            lines_of_code=150,
            cyclomatic_complexity=25,
            cognitive_complexity=18,
            max_nesting_depth=4,
            function_count=5,
            class_count=2,
            efferent_coupling=2,
            afferent_coupling=1,
            smells=[
                SmellLocation(
                    smell_type="long_method",
                    severity="error",
                    message="Method too long",
                    line=50,
                )
            ],
            imports=["utils.helpers", "models.user"],
        ),
        FileDetail(
            path="utils/helpers.py",
            language="python",
            lines_of_code=100,
            cyclomatic_complexity=8,
            cognitive_complexity=6,
            max_nesting_depth=2,
            function_count=8,
            class_count=0,
            efferent_coupling=1,
            afferent_coupling=2,
            smells=[
                SmellLocation(
                    smell_type="unused_import",
                    severity="info",
                    message="Unused import",
                    line=1,
                )
            ],
            imports=["models.user"],
        ),
        FileDetail(
            path="models/user.py",
            language="python",
            lines_of_code=200,
            cyclomatic_complexity=15,
            cognitive_complexity=35,  # High complexity
            max_nesting_depth=3,
            function_count=2,
            class_count=3,
            efferent_coupling=0,
            afferent_coupling=2,
            smells=[],
            imports=[],
        ),
    ]

    dependencies = DependencyGraph(
        edges=[
            DependencyEdge(
                source="src/main.py", target="utils/helpers.py", import_type="import"
            ),
            DependencyEdge(
                source="src/main.py", target="models/user.py", import_type="import"
            ),
            DependencyEdge(
                source="utils/helpers.py",
                target="models/user.py",
                import_type="from_import",
            ),
            # Create circular dependency: models/user.py -> src/main.py
            DependencyEdge(
                source="models/user.py", target="src/main.py", import_type="import"
            ),
        ],
        circular_dependencies=[
            CyclicDependency(
                cycle=["src/main.py", "models/user.py", "src/main.py"], length=3
            )
        ],
    )

    return AnalysisExport(
        metadata=metadata, summary=summary, files=files, dependencies=dependencies
    )


class TestD3Node:
    """Test D3Node dataclass."""

    def test_node_creation(self) -> None:
        """Test creating a D3Node."""
        node = D3Node(
            id="src/main.py",
            label="main.py",
            module="src",
            loc=150,
            complexity=18.0,
            smell_count=2,
            smell_severity="error",
        )

        assert node.id == "src/main.py"
        assert node.label == "main.py"
        assert node.module == "src"
        assert node.loc == 150
        assert node.complexity == 18.0
        assert node.smell_count == 2
        assert node.smell_severity == "error"

    def test_node_to_dict(self) -> None:
        """Test converting D3Node to dictionary."""
        node = D3Node(
            id="test.py",
            label="test.py",
            module="root",
            loc=50,
            complexity=5.0,
            smell_count=0,
            smell_severity="none",
        )

        node_dict = node.to_dict()

        assert isinstance(node_dict, dict)
        assert node_dict["id"] == "test.py"
        assert node_dict["label"] == "test.py"
        assert node_dict["module"] == "root"
        assert node_dict["loc"] == 50
        assert node_dict["complexity"] == 5.0
        assert node_dict["smell_count"] == 0
        assert node_dict["smell_severity"] == "none"


class TestD3Edge:
    """Test D3Edge dataclass."""

    def test_edge_creation(self) -> None:
        """Test creating a D3Edge."""
        edge = D3Edge(source="a.py", target="b.py", coupling=3, circular=False)

        assert edge.source == "a.py"
        assert edge.target == "b.py"
        assert edge.coupling == 3
        assert edge.circular is False

    def test_edge_to_dict(self) -> None:
        """Test converting D3Edge to dictionary."""
        edge = D3Edge(source="a.py", target="b.py", coupling=1, circular=True)

        edge_dict = edge.to_dict()

        assert isinstance(edge_dict, dict)
        assert edge_dict["source"] == "a.py"
        assert edge_dict["target"] == "b.py"
        assert edge_dict["coupling"] == 1
        assert edge_dict["circular"] is True


class TestCreateNode:
    """Test _create_node function."""

    def test_create_node_from_file(self, sample_file_detail: FileDetail) -> None:
        """Test creating D3Node from FileDetail."""
        node = _create_node(sample_file_detail)

        assert node.id == "src/main.py"
        assert node.label == "main.py"
        assert node.module == "src"
        assert node.loc == 150
        assert node.complexity == 18
        assert node.smell_count == 2
        assert node.smell_severity == "error"  # Worst severity

    def test_create_node_no_smells(self) -> None:
        """Test creating node from file with no smells."""
        file = FileDetail(
            path="clean.py",
            language="python",
            lines_of_code=50,
            cyclomatic_complexity=3,
            cognitive_complexity=3,
            max_nesting_depth=1,
            function_count=2,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
            smells=[],
        )

        node = _create_node(file)

        assert node.smell_count == 0
        assert node.smell_severity == "none"

    def test_create_node_root_module(self) -> None:
        """Test creating node for file in root directory."""
        file = FileDetail(
            path="setup.py",
            language="python",
            lines_of_code=20,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            max_nesting_depth=1,
            function_count=1,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
        )

        node = _create_node(file)

        assert node.label == "setup.py"
        assert node.module == "root"


class TestCalculateWorstSeverity:
    """Test _calculate_worst_severity function."""

    def test_no_smells(self) -> None:
        """Test file with no smells."""
        file = FileDetail(
            path="test.py",
            language="python",
            lines_of_code=10,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            max_nesting_depth=1,
            function_count=1,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
            smells=[],
        )

        severity = _calculate_worst_severity(file)
        assert severity == "none"

    def test_single_smell(self) -> None:
        """Test file with single smell."""
        file = FileDetail(
            path="test.py",
            language="python",
            lines_of_code=10,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            max_nesting_depth=1,
            function_count=1,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
            smells=[
                SmellLocation(
                    smell_type="test", severity="warning", message="test", line=1
                )
            ],
        )

        severity = _calculate_worst_severity(file)
        assert severity == "warning"

    def test_multiple_smells_worst_is_error(self) -> None:
        """Test file with multiple smells, worst is error."""
        file = FileDetail(
            path="test.py",
            language="python",
            lines_of_code=10,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            max_nesting_depth=1,
            function_count=1,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
            smells=[
                SmellLocation(
                    smell_type="test1", severity="info", message="test", line=1
                ),
                SmellLocation(
                    smell_type="test2", severity="warning", message="test", line=2
                ),
                SmellLocation(
                    smell_type="test3", severity="error", message="test", line=3
                ),
            ],
        )

        severity = _calculate_worst_severity(file)
        assert severity == "error"

    def test_multiple_smells_worst_is_warning(self) -> None:
        """Test file with multiple smells, worst is warning."""
        file = FileDetail(
            path="test.py",
            language="python",
            lines_of_code=10,
            cyclomatic_complexity=1,
            cognitive_complexity=1,
            max_nesting_depth=1,
            function_count=1,
            class_count=0,
            efferent_coupling=0,
            afferent_coupling=0,
            smells=[
                SmellLocation(
                    smell_type="test1", severity="info", message="test", line=1
                ),
                SmellLocation(
                    smell_type="test2", severity="warning", message="test", line=2
                ),
            ],
        )

        severity = _calculate_worst_severity(file)
        assert severity == "warning"


class TestExtractCircularPaths:
    """Test _extract_circular_paths function."""

    def test_no_circular_dependencies(self) -> None:
        """Test export with no circular dependencies."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(edges=[], circular_dependencies=[]),
        )

        paths = _extract_circular_paths(export)
        assert len(paths) == 0

    def test_single_circular_dependency(self) -> None:
        """Test extracting edges from single cycle."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(
                edges=[],
                circular_dependencies=[
                    CyclicDependency(cycle=["a.py", "b.py", "a.py"], length=3)
                ],
            ),
        )

        paths = _extract_circular_paths(export)

        # Cycle has 3 elements, so creates 3 edges: a->b, b->a, a->a
        assert len(paths) == 3
        assert ("a.py", "b.py") in paths
        assert ("b.py", "a.py") in paths
        assert ("a.py", "a.py") in paths  # Self-loop from wrapping

    def test_multiple_circular_dependencies(self) -> None:
        """Test extracting edges from multiple cycles."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(
                edges=[],
                circular_dependencies=[
                    CyclicDependency(cycle=["a.py", "b.py", "a.py"], length=3),
                    CyclicDependency(cycle=["c.py", "d.py", "e.py", "c.py"], length=4),
                ],
            ),
        )

        paths = _extract_circular_paths(export)

        # First cycle: a->b, b->a, a->a (3 edges)
        # Second cycle: c->d, d->e, e->c, c->c (4 edges)
        assert len(paths) == 7
        assert ("a.py", "b.py") in paths
        assert ("b.py", "a.py") in paths
        assert ("a.py", "a.py") in paths
        assert ("c.py", "d.py") in paths
        assert ("d.py", "e.py") in paths
        assert ("e.py", "c.py") in paths
        assert ("c.py", "c.py") in paths


class TestCreateEdges:
    """Test _create_edges function."""

    def test_create_edges_no_circular(self) -> None:
        """Test creating edges with no circular dependencies."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(
                edges=[
                    DependencyEdge(source="a.py", target="b.py", import_type="import")
                ],
                circular_dependencies=[],
            ),
        )

        circular_paths: set[tuple[str, str]] = set()
        edges = _create_edges(export, circular_paths)

        assert len(edges) == 1
        assert edges[0].source == "a.py"
        assert edges[0].target == "b.py"
        assert edges[0].coupling == 1
        assert edges[0].circular is False

    def test_create_edges_with_circular(self) -> None:
        """Test creating edges with circular dependencies marked."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(
                edges=[
                    DependencyEdge(source="a.py", target="b.py", import_type="import"),
                    DependencyEdge(source="b.py", target="a.py", import_type="import"),
                ],
                circular_dependencies=[
                    CyclicDependency(cycle=["a.py", "b.py", "a.py"], length=3)
                ],
            ),
        )

        circular_paths = {("a.py", "b.py"), ("b.py", "a.py")}
        edges = _create_edges(export, circular_paths)

        assert len(edges) == 2
        assert all(edge.circular for edge in edges)

    def test_create_edges_multiple_imports_same_pair(self) -> None:
        """Test coupling strength from multiple imports between same files."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(
                edges=[
                    DependencyEdge(source="a.py", target="b.py", import_type="import"),
                    DependencyEdge(
                        source="a.py", target="b.py", import_type="from_import"
                    ),
                    DependencyEdge(source="a.py", target="b.py", import_type="import"),
                ],
                circular_dependencies=[],
            ),
        )

        circular_paths: set[tuple[str, str]] = set()
        edges = _create_edges(export, circular_paths)

        # Should consolidate into single edge with coupling=3
        assert len(edges) == 1
        assert edges[0].coupling == 3


class TestTransformForD3:
    """Test transform_for_d3 function."""

    def test_transform_complete_export(self, sample_export: AnalysisExport) -> None:
        """Test transforming complete export to D3 format."""
        result = transform_for_d3(sample_export)

        # Verify structure
        assert "nodes" in result
        assert "links" in result
        assert "summary" in result

        # Verify nodes
        assert len(result["nodes"]) == 3
        node_ids = {node["id"] for node in result["nodes"]}
        assert "src/main.py" in node_ids
        assert "utils/helpers.py" in node_ids
        assert "models/user.py" in node_ids

        # Verify node properties
        main_node = next(n for n in result["nodes"] if n["id"] == "src/main.py")
        assert main_node["label"] == "main.py"
        assert main_node["module"] == "src"
        assert main_node["loc"] == 150
        assert main_node["complexity"] == 18
        assert main_node["smell_severity"] == "error"

        # Verify links
        assert len(result["links"]) == 4

        # Verify circular edge is marked
        circular_edge = next(
            link
            for link in result["links"]
            if link["source"] == "models/user.py" and link["target"] == "src/main.py"
        )
        assert circular_edge["circular"] is True

        # Verify summary
        assert result["summary"]["total_files"] == 3
        assert result["summary"]["total_functions"] == 15
        assert result["summary"]["circular_dependencies"] == 1

    def test_transform_empty_export(self) -> None:
        """Test transforming empty export."""
        export = AnalysisExport(
            metadata=ExportMetadata(
                generated_at=datetime.now(),
                tool_version="0.19.0",
                project_root="/test",
            ),
            summary=MetricsSummary(
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines=0,
                avg_complexity=0.0,
                avg_cognitive_complexity=0.0,
                avg_nesting_depth=0.0,
                total_smells=0,
            ),
            files=[],
            dependencies=DependencyGraph(edges=[], circular_dependencies=[]),
        )

        result = transform_for_d3(export)

        assert result["nodes"] == []
        assert result["links"] == []
        assert result["summary"]["total_files"] == 0


class TestComplexityClass:
    """Test get_complexity_class function."""

    def test_complexity_low(self) -> None:
        """Test low complexity classification."""
        assert get_complexity_class(0) == "low"
        assert get_complexity_class(3) == "low"
        assert get_complexity_class(5) == "low"

    def test_complexity_moderate(self) -> None:
        """Test moderate complexity classification."""
        assert get_complexity_class(6) == "moderate"
        assert get_complexity_class(8) == "moderate"
        assert get_complexity_class(10) == "moderate"

    def test_complexity_high(self) -> None:
        """Test high complexity classification."""
        assert get_complexity_class(11) == "high"
        assert get_complexity_class(15) == "high"
        assert get_complexity_class(20) == "high"

    def test_complexity_very_high(self) -> None:
        """Test very high complexity classification."""
        assert get_complexity_class(21) == "very-high"
        assert get_complexity_class(25) == "very-high"
        assert get_complexity_class(30) == "very-high"

    def test_complexity_critical(self) -> None:
        """Test critical complexity classification."""
        assert get_complexity_class(31) == "critical"
        assert get_complexity_class(50) == "critical"
        assert get_complexity_class(100) == "critical"


class TestSmellClass:
    """Test get_smell_class function."""

    def test_smell_none(self) -> None:
        """Test none severity class."""
        assert get_smell_class("none") == "smell-none"

    def test_smell_info(self) -> None:
        """Test info severity class."""
        assert get_smell_class("info") == "smell-info"

    def test_smell_warning(self) -> None:
        """Test warning severity class."""
        assert get_smell_class("warning") == "smell-warning"

    def test_smell_error(self) -> None:
        """Test error severity class."""
        assert get_smell_class("error") == "smell-error"

    def test_smell_critical(self) -> None:
        """Test critical severity class."""
        assert get_smell_class("critical") == "smell-critical"
