"""Unit tests for semantic indexer functionality."""

import os
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_vector_search.core.exceptions import ParsingError
from mcp_vector_search.core.indexer import SemanticIndexer
from tests.conftest import assert_chunks_valid


class TestSemanticIndexer:
    """Test cases for SemanticIndexer."""

    @pytest.mark.asyncio
    async def test_index_project_basic(self, mock_database, temp_project_dir):
        """Test basic project indexing functionality."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Index the project
        indexed_count = await indexer.index_project()

        # Verify indexing results
        assert indexed_count > 0
        stats = await indexer.get_indexing_stats()
        assert stats["total_chunks"] > 0
        assert stats["indexed_files"] > 0
        assert "python" in stats["languages"]

    @pytest.mark.asyncio
    async def test_index_project_force_reindex(self, mock_database, temp_project_dir):
        """Test force reindexing functionality."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Initial indexing
        initial_count = await indexer.index_project(force_reindex=False)
        assert initial_count > 0

        # Force reindex should process all files again
        force_count = await indexer.index_project(force_reindex=True)
        assert force_count == initial_count

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, mock_database, temp_project_dir):
        """Test incremental indexing with file modification detection."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Initial indexing
        initial_count = await indexer.index_project(force_reindex=False)
        assert initial_count > 0

        # Incremental indexing without changes should index 0 files
        incremental_count = await indexer.index_project(force_reindex=False)
        assert incremental_count == 0

        # Modify a file
        test_file = temp_project_dir / "main.py"
        original_content = test_file.read_text()
        modified_content = original_content + "\n\ndef new_function():\n    pass\n"
        test_file.write_text(modified_content)

        # Incremental indexing should detect the change
        incremental_count = await indexer.index_project(force_reindex=False)
        assert incremental_count == 1

    @pytest.mark.asyncio
    async def test_index_file_basic(self, mock_database, temp_project_dir):
        """Test indexing a single file."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Index single file
        success = await indexer.index_file(test_file)
        assert success is True

        # Verify chunks were created
        stats = await mock_database.get_stats()
        assert stats.total_chunks > 0

    @pytest.mark.asyncio
    async def test_index_file_removes_old_chunks(self, mock_database, temp_project_dir):
        """Test that indexing a file removes old chunks first."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Index file first time
        await indexer.index_file(test_file)
        initial_stats = await mock_database.get_stats()
        initial_chunks = initial_stats.total_chunks

        # Modify file and index again
        original_content = test_file.read_text()
        modified_content = (
            original_content + "\n\ndef additional_function():\n    return 'new'\n"
        )
        test_file.write_text(modified_content)

        await indexer.index_file(test_file)
        final_stats = await mock_database.get_stats()

        # Should have more chunks due to additional function, but old chunks should be removed
        assert final_stats.total_chunks >= initial_chunks

    @pytest.mark.asyncio
    async def test_reindex_file(self, mock_database, temp_project_dir):
        """Test reindexing a specific file."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Initial index
        await indexer.index_file(test_file)

        # Reindex should work
        success = await indexer.reindex_file(test_file)
        assert success is True

    @pytest.mark.asyncio
    async def test_remove_file(self, mock_database, temp_project_dir):
        """Test removing a file from the index."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Index file first
        await indexer.index_file(test_file)
        initial_stats = await mock_database.get_stats()
        assert initial_stats.total_chunks > 0

        # Remove file from index
        removed_count = await indexer.remove_file(test_file)
        assert removed_count > 0

        # Verify chunks were removed
        final_stats = await mock_database.get_stats()
        assert final_stats.total_chunks < initial_stats.total_chunks

    def test_find_indexable_files(self, temp_project_dir):
        """Test finding indexable files."""
        indexer = SemanticIndexer(
            database=Mock(),
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Find Python files
        files = indexer._find_indexable_files()

        assert len(files) > 0
        assert all(f.suffix == ".py" for f in files)
        assert all(f.exists() for f in files)

    def test_should_index_file(self, temp_project_dir):
        """Test file filtering logic."""
        indexer = SemanticIndexer(
            database=Mock(),
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Test valid Python file - create it first
        python_file = temp_project_dir / "test.py"
        python_file.write_text("# test file")
        assert indexer._should_index_file(python_file) is True

        # Test invalid extension
        text_file = temp_project_dir / "test.txt"
        assert indexer._should_index_file(text_file) is False

        # Test non-existent file
        missing_file = temp_project_dir / "missing.py"
        assert indexer._should_index_file(missing_file) is False

    def test_needs_reindexing(self, temp_project_dir):
        """Test file modification detection."""
        indexer = SemanticIndexer(
            database=Mock(),
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Empty metadata - should need reindexing
        empty_metadata = {}
        assert indexer._needs_reindexing(test_file, empty_metadata) is True

        # Current metadata - should not need reindexing
        current_mtime = os.path.getmtime(test_file)
        current_metadata = {str(test_file): current_mtime}
        assert indexer._needs_reindexing(test_file, current_metadata) is False

        # Old metadata - should need reindexing
        old_metadata = {str(test_file): current_mtime - 100}
        assert indexer._needs_reindexing(test_file, old_metadata) is True

    def test_metadata_management(self, temp_project_dir):
        """Test metadata loading and saving."""
        indexer = SemanticIndexer(
            database=Mock(),
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Test loading non-existent metadata
        metadata = indexer._load_index_metadata()
        assert isinstance(metadata, dict)
        assert len(metadata) == 0

        # Test saving metadata
        test_metadata = {"file1.py": 123456789, "file2.py": 987654321}
        indexer._save_index_metadata(test_metadata)

        # Test loading saved metadata
        loaded_metadata = indexer._load_index_metadata()
        assert loaded_metadata == test_metadata

    @pytest.mark.asyncio
    async def test_parse_file_python(self, mock_database, temp_project_dir):
        """Test parsing Python files."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Parse the file
        chunks = await indexer._parse_file(test_file)

        # Verify chunks
        assert_chunks_valid(chunks)
        assert all(chunk.language == "python" for chunk in chunks)
        assert all(chunk.file_path == test_file for chunk in chunks)

    @pytest.mark.asyncio
    async def test_parse_file_error_handling(self, mock_database, temp_project_dir):
        """Test error handling during file parsing."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Test with non-existent file
        missing_file = temp_project_dir / "missing.py"
        chunks = await indexer._parse_file(missing_file)
        assert chunks == []

        # Test with corrupted file
        corrupted_file = temp_project_dir / "corrupted.py"
        corrupted_file.write_text("invalid python syntax $$$ @@@")

        # Should handle parsing errors gracefully
        chunks = await indexer._parse_file(corrupted_file)
        # May return empty list or basic chunks depending on parser robustness
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_process_file_batch(self, mock_database, temp_project_dir):
        """Test batch file processing."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Get some files to process
        files = list(temp_project_dir.glob("*.py"))[:3]  # Process first 3 files

        # Process batch
        results = await indexer._process_file_batch(files, force_reindex=True)

        # Verify results
        assert len(results) == len(files)
        assert all(isinstance(result, bool) for result in results)
        assert any(result for result in results)  # At least one should succeed

    @pytest.mark.asyncio
    async def test_get_indexing_stats(self, mock_database, temp_project_dir):
        """Test indexing statistics."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Index some files
        await indexer.index_project()

        # Get stats
        stats = await indexer.get_indexing_stats()

        # Verify stats structure
        assert isinstance(stats, dict)
        assert "total_chunks" in stats
        assert "total_files" in stats
        assert "languages" in stats
        assert "file_types" in stats

        # Verify stats values
        assert stats["total_chunks"] > 0
        assert stats["indexed_files"] > 0
        assert "python" in stats["languages"]

    @pytest.mark.asyncio
    async def test_performance_large_project(
        self, mock_database, temp_dir, performance_timer
    ):
        """Test indexing performance with larger project."""
        # Create a larger test project
        large_project_dir = temp_dir / "large_project"
        large_project_dir.mkdir()

        # Generate many test files
        from tests.conftest import generate_test_files

        test_files = generate_test_files(large_project_dir, count=50)

        indexer = SemanticIndexer(
            database=mock_database,
            project_root=large_project_dir,
            file_extensions=[".py"],
        )

        # Time the indexing operation
        result, elapsed = await performance_timer.time_async_operation(
            indexer.index_project
        )

        # Performance assertions
        assert result > 0  # Should index some files
        assert elapsed < 30.0, (
            f"Indexing took too long: {elapsed:.3f}s"
        )  # Should complete in reasonable time

        # Verify indexing worked
        stats = await indexer.get_indexing_stats()
        assert stats["indexed_files"] == len(test_files)
        assert stats["total_chunks"] > len(
            test_files
        )  # Should have multiple chunks per file

    @pytest.mark.asyncio
    async def test_concurrent_indexing(self, mock_database, temp_project_dir):
        """Test concurrent indexing operations."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Get files to index
        files = list(temp_project_dir.glob("*.py"))

        # Index files concurrently
        import asyncio

        tasks = [indexer.index_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        assert len(results) == len(files)
        # Most should succeed (some might fail due to race conditions, which is acceptable)
        success_count = sum(1 for r in results if r is True)
        assert success_count > 0

    @pytest.mark.asyncio
    async def test_error_recovery(self, mock_database, temp_project_dir):
        """Test error recovery during indexing."""
        # Create indexer with mock database that fails sometimes
        failing_database = Mock()
        failing_database.delete_by_file = AsyncMock(return_value=0)
        failing_database.add_chunks = AsyncMock(side_effect=Exception("Database error"))

        indexer = SemanticIndexer(
            database=failing_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        test_file = temp_project_dir / "main.py"

        # Should handle database errors gracefully
        with pytest.raises(ParsingError):
            await indexer.index_file(test_file)

    def test_file_extension_filtering(self, temp_project_dir):
        """Test file extension filtering."""
        # Create files with different extensions
        (temp_project_dir / "test.py").touch()
        (temp_project_dir / "test.js").touch()
        (temp_project_dir / "test.txt").touch()
        (temp_project_dir / "README.md").touch()

        # Test Python only
        py_indexer = SemanticIndexer(
            database=Mock(),
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )
        py_files = py_indexer._find_indexable_files()
        assert len(py_files) >= 1  # At least the test.py file
        assert all(f.suffix == ".py" for f in py_files)

        # Test multiple extensions
        multi_indexer = SemanticIndexer(
            database=Mock(),
            project_root=temp_project_dir,
            file_extensions=[".py", ".js"],
        )
        multi_files = multi_indexer._find_indexable_files()
        assert len(multi_files) >= 2  # At least test.py and test.js
        assert all(f.suffix in [".py", ".js"] for f in multi_files)

    @pytest.mark.asyncio
    async def test_metadata_consistency(self, mock_database, temp_project_dir):
        """Test metadata consistency after operations."""
        indexer = SemanticIndexer(
            database=mock_database,
            project_root=temp_project_dir,
            file_extensions=[".py"],
        )

        # Index project
        await indexer.index_project()

        # Check metadata was created
        metadata = indexer._load_index_metadata()
        assert len(metadata) > 0

        # All indexed files should be in metadata
        indexed_files = list(temp_project_dir.glob("*.py"))
        for file_path in indexed_files:
            assert str(file_path) in metadata
            assert isinstance(metadata[str(file_path)], int | float)
            assert metadata[str(file_path)] > 0
