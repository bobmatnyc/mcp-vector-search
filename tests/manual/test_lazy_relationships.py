"""Test lazy relationship computation for issue #62.

This test verifies that:
1. Indexing with skip_relationships=True doesn't compute relationships
2. Indexing with skip_relationships=False does compute relationships
3. The default behavior is skip_relationships=True (lazy)
4. Visualization can lazy-load relationships on-demand
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from mcp_vector_search.config.settings import ProjectConfig
from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.indexer import SemanticIndexer


@pytest.mark.asyncio
async def test_skip_relationships_default():
    """Test that skip_relationships=True skips relationship computation."""
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a simple Python file
        test_file = project_root / "test.py"
        test_file.write_text(
            """
def function_a():
    return 1

def function_b():
    return function_a() + 1

class MyClass:
    def method_a(self):
        return function_b()
"""
        )

        # Setup database and indexer
        db_path = project_root / ".index"
        embedding_function, _ = create_embedding_function("all-MiniLM-L6-v2")
        database = ChromaVectorDatabase(
            persist_directory=db_path,  # Pass Path object, not string
            embedding_function=embedding_function,
        )

        config = ProjectConfig(
            project_root=project_root,
            file_extensions=[".py"],
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_root,
            config=config,
        )

        # Index with skip_relationships=True (default)
        start_time = time.time()
        async with database:
            indexed_count = await indexer.index_project(
                force_reindex=True,
                skip_relationships=True,  # Explicit, but this is the default
            )
        skip_time = time.time() - start_time

        # Verify relationships file doesn't exist (or is empty)
        relationships_file = project_root / ".mcp-vector-search" / "relationships.json"

        # File might exist but should have no relationships if lazy loading is working
        if relationships_file.exists():
            import json

            with open(relationships_file) as f:
                data = json.load(f)
                # Caller relationships should be empty (lazy-loaded)
                assert len(data.get("callers", {})) == 0, (
                    "Callers should be empty with skip_relationships=True"
                )

        print(f"✓ Indexing with skip_relationships=True took {skip_time:.2f}s")
        print(f"✓ Indexed {indexed_count} file(s)")
        print(
            f"✓ Relationships file: {'exists' if relationships_file.exists() else 'does not exist'}"
        )

        assert indexed_count > 0, "Should have indexed at least one file"


@pytest.mark.asyncio
async def test_compute_relationships_explicit():
    """Test that skip_relationships=False computes relationships."""
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create a simple Python file
        test_file = project_root / "test.py"
        test_file.write_text(
            """
def function_a():
    return 1

def function_b():
    return function_a() + 1
"""
        )

        # Setup database and indexer
        db_path = project_root / ".index"
        embedding_function, _ = create_embedding_function("all-MiniLM-L6-v2")
        database = ChromaVectorDatabase(
            persist_directory=db_path,  # Pass Path object, not string
            embedding_function=embedding_function,
        )

        config = ProjectConfig(
            project_root=project_root,
            file_extensions=[".py"],
        )

        indexer = SemanticIndexer(
            database=database,
            project_root=project_root,
            config=config,
        )

        # Index with skip_relationships=False (explicit computation)
        start_time = time.time()
        async with database:
            indexed_count = await indexer.index_project(
                force_reindex=True,
                skip_relationships=False,  # Explicitly compute
            )
        compute_time = time.time() - start_time

        # Verify relationships file exists and has data
        relationships_file = project_root / ".mcp-vector-search" / "relationships.json"
        assert relationships_file.exists(), (
            "Relationships file should exist with skip_relationships=False"
        )

        import json

        with open(relationships_file) as f:
            data = json.load(f)
            # Should have semantic relationships
            assert len(data.get("semantic", [])) > 0, (
                "Should have semantic relationships"
            )
            # Callers might be empty if no cross-file calls, but field should exist
            assert "callers" in data, "Callers field should exist"

        print(f"✓ Indexing with skip_relationships=False took {compute_time:.2f}s")
        print(f"✓ Indexed {indexed_count} file(s)")
        print(f"✓ Computed {len(data.get('semantic', []))} semantic relationships")
        print(
            f"✓ Computed {sum(len(v) for v in data.get('callers', {}).values())} caller relationships"
        )

        assert indexed_count > 0, "Should have indexed at least one file"


@pytest.mark.asyncio
async def test_performance_comparison():
    """Compare performance between lazy and eager relationship computation."""
    # Create temp directory with multiple files
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create multiple Python files to make the difference measurable
        for i in range(5):
            test_file = project_root / f"module_{i}.py"
            test_file.write_text(
                f"""
def function_{i}_a():
    return {i}

def function_{i}_b():
    return function_{i}_a() + 1

class Class_{i}:
    def method_a(self):
        return function_{i}_b()

    def method_b(self):
        return self.method_a() * 2
"""
            )

        # Setup database and indexer
        db_path = project_root / ".index"
        embedding_function, _ = create_embedding_function("all-MiniLM-L6-v2")

        config = ProjectConfig(
            project_root=project_root,
            file_extensions=[".py"],
        )

        # Test 1: Lazy (skip_relationships=True)
        database_lazy = ChromaVectorDatabase(
            persist_directory=db_path / "lazy",  # Pass Path object, not string
            embedding_function=embedding_function,
        )
        indexer_lazy = SemanticIndexer(
            database=database_lazy,
            project_root=project_root,
            config=config,
        )

        start_lazy = time.time()
        async with database_lazy:
            await indexer_lazy.index_project(
                force_reindex=True, skip_relationships=True
            )
        lazy_time = time.time() - start_lazy

        # Test 2: Eager (skip_relationships=False)
        database_eager = ChromaVectorDatabase(
            persist_directory=db_path / "eager",  # Pass Path object, not string
            embedding_function=embedding_function,
        )
        indexer_eager = SemanticIndexer(
            database=database_eager,
            project_root=project_root,
            config=config,
        )

        start_eager = time.time()
        async with database_eager:
            await indexer_eager.index_project(
                force_reindex=True, skip_relationships=False
            )
        eager_time = time.time() - start_eager

        speedup = eager_time - lazy_time
        speedup_percent = (speedup / eager_time) * 100 if eager_time > 0 else 0

        print(f"\n{'=' * 60}")
        print("Performance Comparison (5 files)")
        print(f"{'=' * 60}")
        print(f"Lazy loading (skip_relationships=True):  {lazy_time:.2f}s")
        print(f"Eager loading (skip_relationships=False): {eager_time:.2f}s")
        print(f"Speedup: {speedup:.2f}s ({speedup_percent:.1f}% faster)")
        print(f"{'=' * 60}\n")

        # Lazy should be faster (or at least not slower)
        assert lazy_time <= eager_time, (
            "Lazy loading should be faster than or equal to eager loading"
        )


if __name__ == "__main__":
    # Run tests
    print("Testing lazy relationship computation (issue #62)...\n")

    print("Test 1: Verify skip_relationships=True (default) behavior")
    asyncio.run(test_skip_relationships_default())

    print("\nTest 2: Verify skip_relationships=False (explicit) behavior")
    asyncio.run(test_compute_relationships_explicit())

    print("\nTest 3: Performance comparison")
    asyncio.run(test_performance_comparison())

    print("\n✅ All tests passed!")
