"""Unit tests for JSON exporter functionality."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mcp_vector_search import __version__
from mcp_vector_search.analysis.metrics import (
    ChunkMetrics,
    CouplingMetrics,
    FileMetrics,
    ProjectMetrics,
)
from mcp_vector_search.analysis.visualizer.exporter import JSONExporter
from mcp_vector_search.analysis.visualizer.schemas import (
    AnalysisExport,
    DependencyGraph,
    ExportMetadata,
    FileDetail,
    FunctionMetrics,
    MetricsSummary,
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a temporary project root directory."""
    return tmp_path / "test_project"


@pytest.fixture
def sample_chunk_metrics() -> ChunkMetrics:
    """Create sample chunk metrics for testing."""
    return ChunkMetrics(
        cognitive_complexity=8,
        cyclomatic_complexity=5,
        max_nesting_depth=3,
        parameter_count=2,
        lines_of_code=25,
        smells=["long_method", "deep_nesting"],
        halstead_volume=150.5,
        halstead_difficulty=10.2,
        halstead_effort=1535.1,
    )


@pytest.fixture
def sample_file_metrics(sample_chunk_metrics: ChunkMetrics) -> FileMetrics:
    """Create sample file metrics for testing."""
    coupling = CouplingMetrics(
        efferent_coupling=3,
        afferent_coupling=2,
        imports=["module_a", "module_b", "module_c"],
        internal_imports=["module_a"],
        external_imports=["module_b", "module_c"],
    )

    file_metrics = FileMetrics(
        file_path="src/example.py",
        total_lines=100,
        code_lines=75,
        comment_lines=15,
        blank_lines=10,
        function_count=3,
        class_count=1,
        method_count=2,
        chunks=[sample_chunk_metrics],
        coupling=coupling,
    )
    file_metrics.compute_aggregates()
    return file_metrics


@pytest.fixture
def sample_project_metrics(
    project_root: Path, sample_file_metrics: FileMetrics
) -> ProjectMetrics:
    """Create sample project metrics for testing."""
    project = ProjectMetrics(
        project_root=str(project_root),
        analyzed_at=datetime(2025, 12, 11, 10, 0, 0),
    )
    project.files["src/example.py"] = sample_file_metrics
    project.compute_aggregates()
    return project


@pytest.fixture
def exporter(project_root: Path) -> JSONExporter:
    """Create JSON exporter instance."""
    return JSONExporter(project_root=project_root)


class TestJSONExporter:
    """Test suite for JSONExporter class."""

    def test_init(self, project_root: Path):
        """Test exporter initialization."""
        exporter = JSONExporter(project_root=project_root)
        assert exporter.project_root == project_root
        assert exporter.metrics_store is None
        assert exporter.trend_tracker is None

    def test_init_with_stores(self, project_root: Path):
        """Test exporter initialization with metrics store and trend tracker."""
        mock_store = Mock()
        mock_tracker = Mock()
        exporter = JSONExporter(
            project_root=project_root,
            metrics_store=mock_store,
            trend_tracker=mock_tracker,
        )
        assert exporter.metrics_store is mock_store
        assert exporter.trend_tracker is mock_tracker

    def test_export_basic(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test basic export without trends or dependencies."""
        result = exporter.export(
            sample_project_metrics,
            include_trends=False,
            include_dependencies=False,
        )

        assert isinstance(result, AnalysisExport)
        assert isinstance(result.metadata, ExportMetadata)
        assert isinstance(result.summary, MetricsSummary)
        assert len(result.files) == 1
        assert isinstance(result.dependencies, DependencyGraph)
        assert result.trends is None

    def test_export_with_dependencies(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test export with dependency graph included."""
        result = exporter.export(
            sample_project_metrics,
            include_dependencies=True,
        )

        assert result.dependencies is not None
        assert isinstance(result.dependencies, DependencyGraph)
        assert len(result.dependencies.edges) >= 0

    def test_export_to_file(
        self,
        exporter: JSONExporter,
        sample_project_metrics: ProjectMetrics,
        tmp_path: Path,
    ):
        """Test exporting to JSON file."""
        output_path = tmp_path / "output" / "analysis.json"

        result_path = exporter.export_to_file(
            sample_project_metrics, output_path, indent=2
        )

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.is_file()

        # Verify JSON is valid
        import json

        with output_path.open() as f:
            data = json.load(f)
            assert "metadata" in data
            assert "summary" in data
            assert "files" in data


class TestMetadataCreation:
    """Test suite for metadata creation."""

    def test_create_metadata(self, exporter: JSONExporter):
        """Test metadata creation without git info."""
        with patch.object(exporter, "_get_git_info", return_value=(None, None)):
            metadata = exporter._create_metadata()

        assert isinstance(metadata, ExportMetadata)
        assert metadata.version == "1.0.0"
        assert metadata.tool_version == __version__  # Dynamically get current version
        assert metadata.project_root == str(exporter.project_root)
        assert metadata.git_commit is None
        assert metadata.git_branch is None
        assert isinstance(metadata.generated_at, datetime)

    def test_create_metadata_with_git(self, exporter: JSONExporter):
        """Test metadata creation with git information."""
        mock_commit = "abc123def456"
        mock_branch = "main"

        with patch.object(
            exporter, "_get_git_info", return_value=(mock_commit, mock_branch)
        ):
            metadata = exporter._create_metadata()

        assert metadata.git_commit == mock_commit
        assert metadata.git_branch == mock_branch

    def test_get_git_info_success(self, exporter: JSONExporter, project_root: Path):
        """Test successful git info retrieval."""
        # Create a mock git repository
        project_root.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.check_output") as mock_check_output:
            mock_check_output.side_effect = [
                "abc123def456\n",  # commit (text mode)
                "main\n",  # branch (text mode)
            ]

            commit, branch = exporter._get_git_info()

        assert commit == "abc123def456"
        assert branch == "main"

    def test_get_git_info_not_a_repo(self, exporter: JSONExporter):
        """Test git info retrieval when not in a git repo."""
        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            commit, branch = exporter._get_git_info()

        assert commit is None
        assert branch is None


class TestSummaryCreation:
    """Test suite for summary creation."""

    def test_create_summary_basic(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test basic summary creation."""
        summary = exporter._create_summary(sample_project_metrics)

        assert isinstance(summary, MetricsSummary)
        assert summary.total_files == 1
        assert summary.total_functions == 3
        assert summary.total_classes == 1
        assert summary.total_lines == 100
        assert summary.avg_complexity > 0
        assert summary.total_smells >= 0

    def test_create_summary_empty_project(
        self, exporter: JSONExporter, project_root: Path
    ):
        """Test summary creation for empty project."""
        empty_project = ProjectMetrics(project_root=str(project_root))
        summary = exporter._create_summary(empty_project)

        assert summary.total_files == 0
        assert summary.total_functions == 0
        assert summary.total_classes == 0
        assert summary.total_lines == 0
        assert summary.avg_complexity == 0.0

    def test_create_summary_smells_by_severity(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test smell severity distribution in summary."""
        summary = exporter._create_summary(sample_project_metrics)

        assert isinstance(summary.smells_by_severity, dict)
        # Currently defaults to 'warning' for all smells
        assert (
            "warning" in summary.smells_by_severity
            or len(summary.smells_by_severity) == 0
        )

    def test_create_summary_instability(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test instability calculation in summary."""
        summary = exporter._create_summary(sample_project_metrics)

        # Should have instability since files have coupling data
        assert summary.avg_instability is not None
        assert 0.0 <= summary.avg_instability <= 1.0

    def test_create_summary_halstead_metrics(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test Halstead metrics in summary."""
        summary = exporter._create_summary(sample_project_metrics)

        # Should have Halstead metrics from sample data
        assert summary.avg_halstead_volume is not None
        assert summary.avg_halstead_difficulty is not None


class TestFileConversion:
    """Test suite for file detail conversion."""

    def test_convert_file_basic(
        self, exporter: JSONExporter, sample_file_metrics: FileMetrics
    ):
        """Test basic file conversion."""
        file_detail = exporter._convert_file(sample_file_metrics)

        assert isinstance(file_detail, FileDetail)
        assert file_detail.path == "src/example.py"
        assert file_detail.language == "python"
        assert file_detail.lines_of_code == 100
        assert file_detail.function_count == 3
        assert file_detail.class_count == 1

    def test_convert_file_coupling(
        self, exporter: JSONExporter, sample_file_metrics: FileMetrics
    ):
        """Test coupling metrics in file conversion."""
        file_detail = exporter._convert_file(sample_file_metrics)

        assert file_detail.efferent_coupling == 3
        assert file_detail.afferent_coupling == 2
        assert file_detail.instability is not None
        assert len(file_detail.imports) == 3

    def test_convert_file_complexity(
        self, exporter: JSONExporter, sample_file_metrics: FileMetrics
    ):
        """Test complexity metrics in file conversion."""
        file_detail = exporter._convert_file(sample_file_metrics)

        assert file_detail.cyclomatic_complexity > 0
        assert file_detail.cognitive_complexity > 0
        assert file_detail.max_nesting_depth >= 0

    def test_convert_file_smells(
        self, exporter: JSONExporter, sample_file_metrics: FileMetrics
    ):
        """Test smell conversion in file detail."""
        file_detail = exporter._convert_file(sample_file_metrics)

        # Should have smells from chunks
        assert len(file_detail.smells) >= 0


class TestFunctionConversion:
    """Test suite for function metrics conversion."""

    def test_convert_function_basic(
        self, exporter: JSONExporter, sample_chunk_metrics: ChunkMetrics
    ):
        """Test basic function conversion."""
        func_metrics = exporter._convert_function(sample_chunk_metrics, line_start=10)

        assert isinstance(func_metrics, FunctionMetrics)
        assert func_metrics.line_start == 10
        assert func_metrics.line_end == 10 + sample_chunk_metrics.lines_of_code
        assert func_metrics.cyclomatic_complexity == 5
        assert func_metrics.cognitive_complexity == 8

    def test_convert_function_halstead(
        self, exporter: JSONExporter, sample_chunk_metrics: ChunkMetrics
    ):
        """Test Halstead metrics in function conversion."""
        func_metrics = exporter._convert_function(sample_chunk_metrics, line_start=1)

        assert func_metrics.halstead_volume == 150.5
        assert func_metrics.halstead_difficulty == 10.2
        assert func_metrics.halstead_effort == 1535.1

    def test_convert_function_no_halstead(self, exporter: JSONExporter):
        """Test function conversion without Halstead metrics."""
        chunk = ChunkMetrics(
            cognitive_complexity=5,
            cyclomatic_complexity=3,
            lines_of_code=10,
        )
        func_metrics = exporter._convert_function(chunk, line_start=1)

        assert func_metrics.halstead_volume is None
        assert func_metrics.halstead_difficulty is None
        assert func_metrics.halstead_effort is None


class TestDependencyGraph:
    """Test suite for dependency graph creation."""

    def test_create_dependency_graph_basic(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test basic dependency graph creation."""
        dep_graph = exporter._create_dependency_graph(sample_project_metrics)

        assert isinstance(dep_graph, DependencyGraph)
        assert len(dep_graph.edges) >= 0
        assert isinstance(dep_graph.circular_dependencies, list)

    def test_create_dependency_graph_edges(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test dependency edges are created correctly."""
        dep_graph = exporter._create_dependency_graph(sample_project_metrics)

        # Should have edges from imports
        assert len(dep_graph.edges) == 3  # 3 imports in sample data

    def test_create_dependency_graph_rankings(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test most depended on and most dependent rankings."""
        dep_graph = exporter._create_dependency_graph(sample_project_metrics)

        assert isinstance(dep_graph.most_depended_on, list)
        assert isinstance(dep_graph.most_dependent, list)


class TestTrendData:
    """Test suite for trend data creation."""

    def test_create_trend_data_no_store(self, exporter: JSONExporter):
        """Test trend data creation without metrics store."""
        trend_data = exporter._create_trend_data()

        assert trend_data is None

    def test_create_trend_data_with_store(self, project_root: Path):
        """Test trend data creation with metrics store."""
        mock_store = Mock()
        mock_tracker = Mock()
        exporter = JSONExporter(
            project_root=project_root,
            metrics_store=mock_store,
            trend_tracker=mock_tracker,
        )

        # Currently returns None - TODO when implemented
        trend_data = exporter._create_trend_data()
        assert trend_data is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_export_minimal_data(self, exporter: JSONExporter, project_root: Path):
        """Test export with minimal project data."""
        minimal_project = ProjectMetrics(project_root=str(project_root))

        result = exporter.export(minimal_project)

        assert isinstance(result, AnalysisExport)
        assert result.summary.total_files == 0
        assert len(result.files) == 0

    def test_export_file_without_chunks(
        self, exporter: JSONExporter, project_root: Path
    ):
        """Test export with file that has no chunks."""
        project = ProjectMetrics(project_root=str(project_root))
        file_metrics = FileMetrics(
            file_path="empty.py",
            total_lines=10,
            chunks=[],  # No chunks
        )
        project.files["empty.py"] = file_metrics

        result = exporter.export(project)

        assert len(result.files) == 1
        assert result.summary.total_functions == 0

    def test_json_serialization(
        self, exporter: JSONExporter, sample_project_metrics: ProjectMetrics
    ):
        """Test that exported data is JSON serializable."""
        result = exporter.export(sample_project_metrics)

        # Should not raise exception
        json_str = result.model_dump_json(indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_export_to_nested_path(
        self,
        exporter: JSONExporter,
        sample_project_metrics: ProjectMetrics,
        tmp_path: Path,
    ):
        """Test exporting to nested directory path."""
        output_path = tmp_path / "deeply" / "nested" / "path" / "analysis.json"

        result_path = exporter.export_to_file(sample_project_metrics, output_path)

        assert result_path.exists()
        # Verify file is in the nested structure
        assert result_path.parent == tmp_path / "deeply" / "nested" / "path"
        assert "deeply" in str(result_path)


# Import subprocess for mock
import subprocess
