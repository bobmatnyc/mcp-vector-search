"""Unit tests for pipeline parallelism in index_project()."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vector_search.core.indexer import SemanticIndexer


class TestPipelineParallelism:
    """Test suite for pipeline parallelism feature."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database."""
        db = MagicMock()
        db.embedding_function = MagicMock(return_value=[[0.1] * 768] * 10)
        return db

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory with test files."""
        # Create test files
        (tmp_path / "test1.py").write_text("def foo(): pass")
        (tmp_path / "test2.py").write_text("def bar(): pass")
        (tmp_path / "test3.py").write_text("class Baz: pass")
        return tmp_path

    @pytest.mark.asyncio
    async def test_pipeline_parameter_exists(self, mock_database, temp_project_dir):
        """Test that the pipeline parameter exists with correct default."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        # Check signature
        import inspect

        sig = inspect.signature(indexer.index_project)
        assert "pipeline" in sig.parameters

        # Check default value
        default = sig.parameters["pipeline"].default
        assert default is True, "pipeline parameter should default to True"

    @pytest.mark.asyncio
    async def test_pipeline_mode_enabled_by_default(
        self, mock_database, temp_project_dir
    ):
        """Test that pipeline mode is enabled by default when phase='all'."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ) as mock_pipeline:
            with patch.object(
                indexer, "_phase1_chunk_files", new=AsyncMock(return_value=(3, 10))
            ) as mock_phase1:
                with patch.object(
                    indexer,
                    "_phase2_embed_chunks",
                    new=AsyncMock(return_value=(10, 1)),
                ) as mock_phase2:
                    # Call without explicit pipeline parameter
                    result = await indexer.index_project(phase="all")

                    # Verify pipeline method was called
                    mock_pipeline.assert_called_once()

                    # Verify sequential methods were NOT called
                    mock_phase1.assert_not_called()
                    mock_phase2.assert_not_called()

                    # Verify result
                    assert result == 3  # files indexed

    @pytest.mark.asyncio
    async def test_pipeline_mode_explicit_true(self, mock_database, temp_project_dir):
        """Test that pipeline=True uses pipeline mode."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ) as mock_pipeline:
            result = await indexer.index_project(pipeline=True, phase="all")

            mock_pipeline.assert_called_once()
            assert result == 3

    @pytest.mark.asyncio
    async def test_sequential_mode_when_disabled(self, mock_database, temp_project_dir):
        """Test that pipeline=False uses sequential execution."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ) as mock_pipeline:
            with patch.object(
                indexer, "_phase1_chunk_files", new=AsyncMock(return_value=(3, 10))
            ) as mock_phase1:
                with patch.object(
                    indexer,
                    "_phase2_embed_chunks",
                    new=AsyncMock(return_value=(10, 1)),
                ) as mock_phase2:
                    # Call with pipeline=False
                    result = await indexer.index_project(pipeline=False, phase="all")

                    # Verify pipeline method was NOT called
                    mock_pipeline.assert_not_called()

                    # Verify sequential methods were called
                    mock_phase1.assert_called_once()
                    mock_phase2.assert_called_once()

                    # Verify result
                    assert result == 3

    @pytest.mark.asyncio
    async def test_no_pipeline_for_chunk_phase(self, mock_database, temp_project_dir):
        """Test that pipeline is not used when phase='chunk'."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ) as mock_pipeline:
            with patch.object(
                indexer, "_phase1_chunk_files", new=AsyncMock(return_value=(3, 10))
            ) as mock_phase1:
                # Call with phase='chunk' (pipeline should be ignored)
                result = await indexer.index_project(
                    pipeline=True,
                    phase="chunk",  # pipeline param ignored
                )

                # Pipeline should NOT be called
                mock_pipeline.assert_not_called()

                # Phase 1 should be called
                mock_phase1.assert_called_once()

                assert result == 3

    @pytest.mark.asyncio
    async def test_no_pipeline_for_embed_phase(self, mock_database, temp_project_dir):
        """Test that pipeline is not used when phase='embed'."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ) as mock_pipeline:
            with patch.object(
                indexer, "_phase2_embed_chunks", new=AsyncMock(return_value=(10, 1))
            ) as mock_phase2:
                # Call with phase='embed' (pipeline should be ignored)
                result = await indexer.index_project(
                    pipeline=True,
                    phase="embed",  # pipeline param ignored
                )

                # Pipeline should NOT be called
                mock_pipeline.assert_not_called()

                # Phase 2 should be called
                mock_phase2.assert_called_once()

                # Result is 0 for phase='embed' (no files indexed)
                assert result == 0

    @pytest.mark.asyncio
    async def test_pipeline_method_exists(self, mock_database, temp_project_dir):
        """Test that _index_with_pipeline method exists."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        # Check method exists
        assert hasattr(indexer, "_index_with_pipeline")
        assert callable(indexer._index_with_pipeline)

        # Check signature
        import inspect

        sig = inspect.signature(indexer._index_with_pipeline)
        assert "force_reindex" in sig.parameters

    @pytest.mark.asyncio
    async def test_pipeline_return_type(self, mock_database, temp_project_dir):
        """Test that _index_with_pipeline returns correct tuple."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        # Mock the internal methods
        with patch.object(
            indexer.file_discovery,
            "find_indexable_files",
            return_value=[
                temp_project_dir / "test1.py",
                temp_project_dir / "test2.py",
            ],
        ):
            with patch.object(
                indexer.chunks_backend, "file_changed", return_value=False
            ):
                # Call should return (files_indexed, chunks_created, chunks_embedded)
                result = await indexer._index_with_pipeline(force_reindex=False)

                assert isinstance(result, tuple)
                assert len(result) == 3

                files_indexed, chunks_created, chunks_embedded = result
                assert isinstance(files_indexed, int)
                assert isinstance(chunks_created, int)
                assert isinstance(chunks_embedded, int)

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, mock_database, temp_project_dir):
        """Test that existing code without pipeline parameter still works."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ):
            # Call without pipeline parameter (should use default)
            result = await indexer.index_project(force_reindex=True)

            assert isinstance(result, int)
            assert result == 3

    @pytest.mark.asyncio
    async def test_pipeline_with_force_reindex(self, mock_database, temp_project_dir):
        """Test that pipeline works with force_reindex=True."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer, "_index_with_pipeline", new=AsyncMock(return_value=(3, 10, 10))
        ) as mock_pipeline:
            result = await indexer.index_project(force_reindex=True, pipeline=True)

            # Verify pipeline was called with force_reindex=True
            mock_pipeline.assert_called_once_with(True)
            assert result == 3


class TestPipelineInternals:
    """Test internal implementation details of pipeline."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database with embedding function."""
        db = MagicMock()
        db.embedding_function = MagicMock(return_value=[[0.1] * 768] * 10)
        return db

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory."""
        (tmp_path / "test.py").write_text("def foo(): pass")
        return tmp_path

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self, mock_database, temp_project_dir):
        """Test that pipeline completes successfully with no files."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer.file_discovery, "find_indexable_files", return_value=[]
        ):
            # Run pipeline (should complete quickly with no files)
            result = await indexer._index_with_pipeline(force_reindex=False)

            # Verify it returns the expected tuple
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result == (0, 0, 0)  # No files, no chunks, no embeddings

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_files(self, mock_database, temp_project_dir):
        """Test that pipeline handles empty file list gracefully."""
        indexer = SemanticIndexer(
            database=mock_database, project_root=temp_project_dir, file_extensions=[".py"]
        )

        with patch.object(
            indexer.file_discovery, "find_indexable_files", return_value=[]
        ):
            result = await indexer._index_with_pipeline(force_reindex=False)

            assert result == (0, 0, 0)  # No files, no chunks, no embeddings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
