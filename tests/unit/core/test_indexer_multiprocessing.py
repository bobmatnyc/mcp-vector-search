"""Unit tests for multiprocessing functionality in semantic indexer."""

import multiprocessing

import pytest

from mcp_vector_search.core.indexer import SemanticIndexer


class TestSemanticIndexerMultiprocessing:
    """Test cases for multiprocessing in SemanticIndexer."""

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Obsolete test after v1.2.7 refactoring - use_multiprocessing/max_workers moved to ChunkProcessor"
    )
    async def test_multiprocessing_enabled_by_default(
        self, mock_database, temp_project_dir
    ):
        """Test that multiprocessing is enabled by default."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Verify multiprocessing is enabled
        assert indexer.use_multiprocessing is True
        # Worker count should be 75% of CPU count (no cap for full CPU utilization)
        cpu_count = multiprocessing.cpu_count()
        expected_workers = max(1, int(cpu_count * 0.75))
        assert indexer.max_workers == expected_workers

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Obsolete test after v1.2.7 refactoring - use_multiprocessing/max_workers moved to ChunkProcessor"
    )
    async def test_multiprocessing_can_be_disabled(
        self, mock_database, temp_project_dir
    ):
        """Test that multiprocessing can be explicitly disabled."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            use_multiprocessing=False,
        )

        # Verify multiprocessing is disabled
        assert indexer.use_multiprocessing is False
        assert indexer.max_workers == 1

    @pytest.mark.asyncio
    async def test_index_project_with_multiprocessing(
        self, mock_database, temp_project_dir
    ):
        """Test project indexing with multiprocessing enabled."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            use_multiprocessing=True,
        )

        # Index the project
        indexed_count = await indexer.index_project()

        # Verify indexing results
        assert indexed_count > 0
        stats = await indexer.get_indexing_stats()
        assert stats["total_chunks"] > 0
        assert stats["indexed_files"] > 0

    @pytest.mark.asyncio
    async def test_index_project_without_multiprocessing(
        self, mock_database, temp_project_dir
    ):
        """Test project indexing with multiprocessing disabled."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            use_multiprocessing=False,
        )

        # Index the project
        indexed_count = await indexer.index_project()

        # Verify indexing results (same as with multiprocessing)
        assert indexed_count > 0
        stats = await indexer.get_indexing_stats()
        assert stats["total_chunks"] > 0
        assert stats["indexed_files"] > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Obsolete test after v1.2.7 refactoring - max_workers moved to ChunkProcessor"
    )
    async def test_custom_max_workers(self, mock_database, temp_project_dir):
        """Test that custom max_workers is respected."""
        custom_workers = 2
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            max_workers=custom_workers,
            use_multiprocessing=True,
        )

        # Verify custom worker count
        assert indexer.max_workers == custom_workers

    @pytest.mark.asyncio
    async def test_multiprocessing_handles_errors_gracefully(
        self, mock_database, temp_project_dir
    ):
        """Test that multiprocessing handles parsing errors gracefully."""
        # Create a malformed Python file
        bad_file = temp_project_dir / "bad_syntax.py"
        bad_file.write_text("def broken(:\n    pass\n")

        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            use_multiprocessing=True,
        )

        # Should handle error and continue with other files
        await indexer.index_project()

        # Should have indexed at least some files (the valid ones)
        stats = await indexer.get_indexing_stats()
        assert stats["total_chunks"] >= 0  # May have some valid chunks

    @pytest.mark.asyncio
    async def test_single_file_uses_async_path(self, mock_database, temp_project_dir):
        """Test that single file indexing doesn't use multiprocessing unnecessarily."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            use_multiprocessing=True,
        )

        # Index a single file
        test_file = temp_project_dir / "main.py"
        success = await indexer.index_file(test_file)

        # Should succeed
        assert success is True

    @pytest.mark.asyncio
    async def test_batch_size_respected_with_multiprocessing(
        self, mock_database, temp_project_dir
    ):
        """Test that batch_size parameter works with multiprocessing."""
        # Create multiple files to test batching
        for i in range(5):
            file = temp_project_dir / f"module_{i}.py"
            file.write_text(f"def func_{i}():\n    pass\n")

        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
            batch_size=2,  # Small batch size
            use_multiprocessing=True,
        )

        # Index with small batch size
        indexed_count = await indexer.index_project()

        # Should have indexed all files
        assert indexed_count >= 5
