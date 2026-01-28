"""Tests for metric collector integration with SemanticIndexer."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mcp_vector_search.analysis.collectors.complexity import (
    CognitiveComplexityCollector,
    CyclomaticComplexityCollector,
    MethodCountCollector,
    NestingDepthCollector,
    ParameterCountCollector,
)
from mcp_vector_search.analysis.metrics import ChunkMetrics
from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.indexer import SemanticIndexer
from mcp_vector_search.core.metrics_collector import EXTENSION_TO_LANGUAGE
from mcp_vector_search.core.models import CodeChunk


@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - collectors now handled by ChunkProcessor"
)
class TestIndexerCollectorInitialization:
    """Test collector initialization in SemanticIndexer."""

    def test_default_collectors_created(self, tmp_path):
        """Test that default collectors are created when none provided."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Should have 5 default collectors
        assert len(indexer.collectors) == 5
        assert any(
            isinstance(c, CognitiveComplexityCollector) for c in indexer.collectors
        )
        assert any(
            isinstance(c, CyclomaticComplexityCollector) for c in indexer.collectors
        )
        assert any(isinstance(c, NestingDepthCollector) for c in indexer.collectors)
        assert any(isinstance(c, ParameterCountCollector) for c in indexer.collectors)
        assert any(isinstance(c, MethodCountCollector) for c in indexer.collectors)

    def test_custom_collectors_provided(self, tmp_path):
        """Test that custom collectors can be provided."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])
        custom_collectors = [CognitiveComplexityCollector()]

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
            collectors=custom_collectors,
        )

        # Should use provided collectors
        assert len(indexer.collectors) == 1
        assert isinstance(indexer.collectors[0], CognitiveComplexityCollector)

    def test_empty_collectors_list(self, tmp_path):
        """Test that empty collector list disables metric collection."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
            collectors=[],
        )

        # Should have empty collector list
        assert len(indexer.collectors) == 0


@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - collectors now handled by ChunkProcessor"
)
class TestDefaultCollectors:
    """Test the _default_collectors method."""

    def test_default_collectors_returns_all_complexity_collectors(self, tmp_path):
        """Test that _default_collectors returns all 5 complexity collectors."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        collectors = indexer._default_collectors()

        # Should return 5 collectors
        assert len(collectors) == 5

        # Verify each collector type is present
        collector_types = {type(c) for c in collectors}
        assert CognitiveComplexityCollector in collector_types
        assert CyclomaticComplexityCollector in collector_types
        assert NestingDepthCollector in collector_types
        assert ParameterCountCollector in collector_types
        assert MethodCountCollector in collector_types


@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - collectors now handled by ChunkProcessor"
)
class TestCollectMetrics:
    """Test the _collect_metrics method."""

    def test_collect_metrics_returns_chunk_metrics(self, tmp_path):
        """Test that _collect_metrics returns valid ChunkMetrics."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Create a sample code chunk
        chunk = CodeChunk(
            content="def foo():\n    if x > 0:\n        return x\n    return 0",
            file_path=Path("/test/file.py"),
            start_line=1,
            end_line=4,
            language="python",
            chunk_type="function",
            function_name="foo",
            parameters=[],
        )

        source_code = b"def foo():\n    if x > 0:\n        return x\n    return 0"
        language = "python"

        metrics = indexer._collect_metrics(chunk, source_code, language)

        # Should return ChunkMetrics
        assert isinstance(metrics, ChunkMetrics)
        assert metrics.lines_of_code == 4
        assert metrics.cognitive_complexity >= 0
        assert metrics.cyclomatic_complexity >= 1  # Baseline is 1
        assert metrics.parameter_count == 0

    def test_collect_metrics_with_parameters(self, tmp_path):
        """Test that _collect_metrics correctly counts parameters."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Create a chunk with parameters
        chunk = CodeChunk(
            content="def add(x, y):\n    return x + y",
            file_path=Path("/test/file.py"),
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            function_name="add",
            parameters=[{"name": "x"}, {"name": "y"}],
        )

        source_code = b"def add(x, y):\n    return x + y"
        language = "python"

        metrics = indexer._collect_metrics(chunk, source_code, language)

        # Should correctly count parameters
        assert metrics.parameter_count == 2


@pytest.mark.skip(
    reason="Obsolete tests after v1.2.7 refactoring - collectors now handled by ChunkProcessor"
)
class TestEstimateComplexity:
    """Test complexity estimation helper methods."""

    def test_estimate_cognitive_complexity_simple(self, tmp_path):
        """Test cognitive complexity estimation for simple code."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Simple code with no control flow
        content = "def foo():\n    return 42"
        complexity = indexer._estimate_cognitive_complexity(content)
        assert complexity == 0

    def test_estimate_cognitive_complexity_with_if(self, tmp_path):
        """Test cognitive complexity estimation with if statements."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Code with if statement
        content = "def foo():\n    if x > 0:\n        return x"
        complexity = indexer._estimate_cognitive_complexity(content)
        assert complexity >= 1

    def test_estimate_cyclomatic_complexity_baseline(self, tmp_path):
        """Test cyclomatic complexity starts at 1."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Simple code
        content = "def foo():\n    return 42"
        complexity = indexer._estimate_cyclomatic_complexity(content)
        assert complexity >= 1  # Baseline

    def test_estimate_nesting_depth_flat(self, tmp_path):
        """Test nesting depth estimation for flat code."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Flat code
        content = "def foo():\nreturn 42"
        depth = indexer._estimate_nesting_depth(content)
        assert depth >= 0

    def test_estimate_nesting_depth_nested(self, tmp_path):
        """Test nesting depth estimation for nested code."""
        mock_db = MagicMock()
        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        indexer = SemanticIndexer(
            database=mock_db,
            project_root=tmp_path,
            config=config,
        )

        # Nested code (4 spaces per level)
        content = "def foo():\n    if x:\n        if y:\n            return z"
        depth = indexer._estimate_nesting_depth(content)
        assert depth >= 2  # At least 2 levels of nesting


class TestExtensionToLanguageMapping:
    """Test the EXTENSION_TO_LANGUAGE constant."""

    def test_extension_mapping_contains_common_languages(self):
        """Test that extension mapping contains common languages."""
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".java"] == "java"
        assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"
        assert EXTENSION_TO_LANGUAGE[".php"] == "php"
        assert EXTENSION_TO_LANGUAGE[".rb"] == "ruby"

    def test_extension_mapping_handles_jsx_tsx(self):
        """Test that JSX/TSX map to JavaScript/TypeScript."""
        assert EXTENSION_TO_LANGUAGE[".jsx"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".tsx"] == "typescript"


@pytest.mark.asyncio
class TestIndexFileWithMetrics:
    """Test index_file method with metric collection."""

    async def test_index_file_collects_metrics(self, tmp_path):
        """Test that index_file collects and stores metrics."""
        # Create a test Python file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            "def foo():\n    if x > 0:\n        return x\n    return 0"
        )

        # Mock database
        mock_db = AsyncMock()
        mock_db.delete_by_file = AsyncMock(return_value=0)
        mock_db.add_chunks = AsyncMock()

        # Mock parser
        mock_parser = MagicMock()
        mock_chunk = CodeChunk(
            content="def foo():\n    if x > 0:\n        return x\n    return 0",
            file_path=test_file,
            start_line=1,
            end_line=4,
            language="python",
            chunk_type="function",
            function_name="foo",
            parameters=[],
        )
        mock_parser.parse_file = AsyncMock(return_value=[mock_chunk])

        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        with patch(
            "mcp_vector_search.core.indexer.get_parser_registry"
        ) as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_parser_for_file.return_value = mock_parser
            mock_registry.return_value = mock_registry_instance

            indexer = SemanticIndexer(
                database=mock_db,
                project_root=tmp_path,
                config=config,
            )

            # Index the file
            success = await indexer.index_file(test_file)

            # Should succeed
            assert success

            # Database add_chunks should be called with metrics
            assert mock_db.add_chunks.called
            call_args = mock_db.add_chunks.call_args

            # Check that chunks were passed
            chunks_arg = call_args[0][0]
            assert len(chunks_arg) > 0

            # Check that metrics were passed
            metrics_arg = call_args[1].get("metrics")
            assert metrics_arg is not None
            assert isinstance(metrics_arg, dict)
            assert len(metrics_arg) > 0

            # Verify metrics structure
            for chunk_id, metric_dict in metrics_arg.items():
                assert isinstance(chunk_id, str)
                assert isinstance(metric_dict, dict)
                assert "cognitive_complexity" in metric_dict
                assert "cyclomatic_complexity" in metric_dict
                assert "lines_of_code" in metric_dict

    async def test_index_file_with_no_collectors(self, tmp_path):
        """Test that index_file works without collectors."""
        # Create a test Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42")

        # Mock database
        mock_db = AsyncMock()
        mock_db.delete_by_file = AsyncMock(return_value=0)
        mock_db.add_chunks = AsyncMock()

        # Mock parser
        mock_parser = MagicMock()
        mock_chunk = CodeChunk(
            content="def foo():\n    return 42",
            file_path=test_file,
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            function_name="foo",
        )
        mock_parser.parse_file = AsyncMock(return_value=[mock_chunk])

        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        with patch(
            "mcp_vector_search.core.indexer.get_parser_registry"
        ) as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_parser_for_file.return_value = mock_parser
            mock_registry.return_value = mock_registry_instance

            # Create indexer with no collectors
            indexer = SemanticIndexer(
                database=mock_db,
                project_root=tmp_path,
                config=config,
                collectors=[],  # Disable collectors
            )

            # Index the file
            success = await indexer.index_file(test_file)

            # Should succeed
            assert success

            # Database add_chunks should be called without metrics
            assert mock_db.add_chunks.called
            call_args = mock_db.add_chunks.call_args

            # Check that metrics is None
            metrics_arg = call_args[1].get("metrics")
            assert metrics_arg is None

    @pytest.mark.skip(
        reason="Obsolete test after v1.2.7 refactoring - _collect_metrics moved to MetricsCollector class"
    )
    async def test_index_file_handles_metric_collection_errors(self, tmp_path):
        """Test that index_file handles metric collection errors gracefully."""
        # Create a test Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42")

        # Mock database
        mock_db = AsyncMock()
        mock_db.delete_by_file = AsyncMock(return_value=0)
        mock_db.add_chunks = AsyncMock()

        # Mock parser
        mock_parser = MagicMock()
        mock_chunk = CodeChunk(
            content="def foo():\n    return 42",
            file_path=test_file,
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            function_name="foo",
        )
        mock_parser.parse_file = AsyncMock(return_value=[mock_chunk])

        config = ProjectConfig(project_root=tmp_path, file_extensions=[".py"])

        with patch(
            "mcp_vector_search.core.indexer.get_parser_registry"
        ) as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_parser_for_file.return_value = mock_parser
            mock_registry.return_value = mock_registry_instance

            indexer = SemanticIndexer(
                database=mock_db,
                project_root=tmp_path,
                config=config,
            )

            # Mock _collect_metrics to raise an exception
            indexer._collect_metrics = Mock(side_effect=Exception("Test error"))

            # Index the file - should still succeed but without metrics
            success = await indexer.index_file(test_file)

            # Should succeed (error is logged but not raised)
            assert success

            # Database add_chunks should be called without metrics
            assert mock_db.add_chunks.called
            call_args = mock_db.add_chunks.call_args

            # Check that metrics is None due to error
            metrics_arg = call_args[1].get("metrics")
            assert metrics_arg is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
