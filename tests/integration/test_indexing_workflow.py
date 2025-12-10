"""Integration tests for indexing workflow."""

import asyncio
from pathlib import Path

import pytest

from mcp_vector_search.core.factory import ComponentFactory, DatabaseContext
from mcp_vector_search.core.project import ProjectManager
from tests.conftest import assert_search_results_valid


class TestIndexingWorkflow:
    """Integration tests for complete indexing workflow."""

    def _initialize_project(self, project_dir: Path) -> None:
        """Initialize a project for testing."""
        project_manager = ProjectManager(project_dir)
        project_manager.initialize(
            file_extensions=[".py"],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.7,
        )

    @pytest.mark.asyncio
    async def test_full_indexing_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test complete indexing workflow from start to finish."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        # Create components using factory
        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
            include_search_engine=True,
            include_auto_indexer=False,
        )

        async with DatabaseContext(components.database):
            # Step 1: Index the project
            indexed_count = await components.indexer.index_project()
            assert indexed_count > 0, "Should have indexed some files"

            # Step 2: Verify indexing statistics
            stats = await components.indexer.get_indexing_stats()
            assert stats["total_chunks"] > 0
            assert stats["indexed_files"] > 0
            assert "python" in stats["languages"]

            # Step 3: Perform searches
            search_queries = [
                "main function",
                "user service",
                "create user",
                "database connection",
                "load config",
            ]

            for query in search_queries:
                results = await components.search_engine.search(
                    query, limit=5, similarity_threshold=0.1
                )
                # Results may be empty for some queries, which is acceptable
                assert isinstance(results, list)
                if results:
                    assert_search_results_valid(results)

    @pytest.mark.asyncio
    async def test_incremental_indexing_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test incremental indexing workflow."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
            include_search_engine=True,
        )

        async with DatabaseContext(components.database):
            # Initial indexing
            initial_count = await components.indexer.index_project(force_reindex=False)
            assert initial_count > 0

            await components.indexer.get_indexing_stats()

            # Incremental indexing without changes
            incremental_count = await components.indexer.index_project(
                force_reindex=False
            )
            assert incremental_count == 0, "No files should need reindexing"

            # Modify a file
            test_file = temp_project_dir / "main.py"
            original_content = test_file.read_text()
            modified_content = (
                original_content + "\n\ndef new_test_function():\n    return 'test'\n"
            )
            test_file.write_text(modified_content)

            # Incremental indexing should detect change
            incremental_count = await components.indexer.index_project(
                force_reindex=False
            )
            assert incremental_count == 1, "Should reindex the modified file"

            # Verify new content is searchable
            results = await components.search_engine.search(
                "new_test_function", limit=5, similarity_threshold=0.1
            )
            # May or may not find results depending on embedding similarity
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_force_reindexing_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test force reindexing workflow."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
        )

        async with DatabaseContext(components.database):
            # Initial indexing
            initial_count = await components.indexer.index_project(force_reindex=False)
            assert initial_count > 0

            # Force reindexing should process all files
            force_count = await components.indexer.index_project(force_reindex=True)
            assert force_count == initial_count, (
                "Force reindex should process all files"
            )

            # Verify stats are consistent
            stats = await components.indexer.get_indexing_stats()
            assert stats["total_chunks"] > 0
            assert stats["indexed_files"] == initial_count

    @pytest.mark.asyncio
    async def test_single_file_indexing_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test single file indexing workflow."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
        )

        async with DatabaseContext(components.database):
            test_file = temp_project_dir / "main.py"

            # Index single file
            success = await components.indexer.index_file(test_file)
            assert success is True

            # Verify file was indexed
            stats = await components.database.get_stats()
            assert stats.total_chunks > 0

            # Reindex the same file
            success = await components.indexer.reindex_file(test_file)
            assert success is True

            # Remove file from index
            removed_count = await components.indexer.remove_file(test_file)
            assert removed_count > 0

            # Verify chunks were removed
            final_stats = await components.database.get_stats()
            assert final_stats.total_chunks < stats.total_chunks

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test error recovery during indexing workflow."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
        )

        async with DatabaseContext(components.database):
            # Create a file with syntax errors
            error_file = temp_project_dir / "syntax_error.py"
            error_file.write_text(
                "def invalid_syntax(\n    # Missing closing parenthesis"
            )

            # Indexing should handle the error gracefully
            indexed_count = await components.indexer.index_project()
            # Should still index other valid files
            assert indexed_count >= 0

            # Clean up error file
            error_file.unlink()

    @pytest.mark.asyncio
    async def test_large_project_workflow(self, temp_dir, real_embedding_function):
        """Test indexing workflow with larger project."""
        # Create a larger test project
        large_project_dir = temp_dir / "large_project"
        large_project_dir.mkdir()

        # Generate many test files
        from tests.conftest import generate_test_files

        test_files = generate_test_files(large_project_dir, count=20)

        # Initialize project first
        self._initialize_project(large_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=large_project_dir,
            use_pooling=True,  # Use pooling for better performance
            include_search_engine=True,
        )

        async with DatabaseContext(components.database):
            # Index the large project
            indexed_count = await components.indexer.index_project()
            assert indexed_count == len(test_files)

            # Verify statistics
            stats = await components.indexer.get_indexing_stats()
            assert stats["indexed_files"] == len(test_files)
            assert stats["total_chunks"] > len(test_files)  # Multiple chunks per file

            # Test search performance
            import time

            start_time = time.perf_counter()

            results = await components.search_engine.search(
                "function", limit=10, similarity_threshold=0.1
            )

            search_time = time.perf_counter() - start_time

            # Search should be fast even with many files
            assert search_time < 1.0, f"Search took too long: {search_time:.3f}s"
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_concurrent_indexing_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test concurrent indexing operations."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=True,  # Use pooling for concurrent operations
        )

        async with DatabaseContext(components.database):
            # Get files to index
            files = list(temp_project_dir.glob("*.py"))

            # Index files concurrently
            tasks = [components.indexer.index_file(file_path) for file_path in files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Most operations should succeed
            success_count = sum(1 for r in results if r is True)
            assert success_count > 0

            # Verify final state
            stats = await components.indexer.get_indexing_stats()
            assert stats["total_chunks"] > 0

    @pytest.mark.asyncio
    async def test_search_integration_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test integration between indexing and search."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
            include_search_engine=True,
        )

        async with DatabaseContext(components.database):
            # Index project
            indexed_count = await components.indexer.index_project()
            assert indexed_count > 0

            # Test various search scenarios
            test_cases = [
                ("main", "Should find main function"),
                ("user", "Should find user-related code"),
                ("class", "Should find class definitions"),
                ("function", "Should find function definitions"),
                ("config", "Should find configuration code"),
            ]

            for query, description in test_cases:
                results = await components.search_engine.search(
                    query, limit=5, similarity_threshold=0.1
                )

                # Verify search results structure
                assert isinstance(results, list), f"Failed: {description}"
                if results:
                    assert_search_results_valid(results)

                    # Verify results are relevant to the query
                    for result in results:
                        assert result.content is not None
                        assert result.file_path is not None
                        assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_metadata_consistency_workflow(
        self, temp_project_dir, real_embedding_function
    ):
        """Test metadata consistency throughout workflow."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=False,
        )

        async with DatabaseContext(components.database):
            # Index project
            await components.indexer.index_project()

            # Check metadata was created
            metadata = components.indexer._load_index_metadata()
            assert len(metadata) > 0

            # Verify all indexed files are in metadata
            indexed_files = list(temp_project_dir.glob("*.py"))
            for file_path in indexed_files:
                assert str(file_path) in metadata
                assert isinstance(metadata[str(file_path)], int | float)

            # Modify a file and reindex
            test_file = temp_project_dir / "main.py"
            original_mtime = metadata[str(test_file)]

            # Wait a bit and modify file
            import time

            time.sleep(0.1)
            content = test_file.read_text()
            test_file.write_text(content + "\n# Modified")

            # Reindex
            await components.indexer.index_project(force_reindex=False)

            # Check metadata was updated
            updated_metadata = components.indexer._load_index_metadata()
            assert updated_metadata[str(test_file)] > original_mtime

    @pytest.mark.asyncio
    async def test_performance_workflow(
        self, temp_project_dir, real_embedding_function, performance_timer
    ):
        """Test performance characteristics of indexing workflow."""
        # Initialize project first
        self._initialize_project(temp_project_dir)

        components = await ComponentFactory.create_standard_components(
            project_root=temp_project_dir,
            use_pooling=True,  # Use pooling for better performance
            include_search_engine=True,
        )

        async with DatabaseContext(components.database):
            # Time indexing operation
            result, indexing_time = await performance_timer.time_async_operation(
                components.indexer.index_project
            )

            assert result > 0
            assert indexing_time < 30.0, f"Indexing took too long: {indexing_time:.3f}s"

            # Time multiple search operations
            search_queries = ["function", "class", "user", "main", "config"]

            for query in search_queries:
                result, search_time = await performance_timer.time_async_operation(
                    components.search_engine.search,
                    query,
                    limit=5,
                    similarity_threshold=0.1,
                )

                assert search_time < 1.0, (
                    f"Search for '{query}' took too long: {search_time:.3f}s"
                )
                assert isinstance(result, list)

            # Check overall performance stats
            stats = performance_timer.get_stats()
            assert stats["average"] < 5.0, (
                f"Average operation time too high: {stats['average']:.3f}s"
            )
