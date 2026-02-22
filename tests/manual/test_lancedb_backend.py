"""Manual test for LanceDB backend functionality.

This script tests the LanceDB backend to ensure it works as a drop-in
replacement for ChromaDB.

Usage:
    python tests/manual/test_lancedb_backend.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_vector_search.core.embeddings import create_embedding_function
from mcp_vector_search.core.lancedb_backend import LanceVectorDatabase
from mcp_vector_search.core.models import CodeChunk


async def test_lancedb_basic_operations():
    """Test basic LanceDB operations: add, search, delete, stats."""
    print("🧪 Testing LanceDB Backend\n")

    # Create temporary directory for test database
    test_dir = Path(__file__).parent / "test_lancedb"
    test_dir.mkdir(exist_ok=True)

    print(f"📁 Test directory: {test_dir}")

    # Create embedding function
    print("📦 Loading embedding model...")
    embedding_function, _ = create_embedding_function(
        "jinaai/jina-embeddings-v2-base-code"
    )

    # Create LanceDB instance
    async with LanceVectorDatabase(
        persist_directory=test_dir,
        embedding_function=embedding_function,
        collection_name="test_collection",
    ) as db:
        print("✅ LanceDB initialized\n")

        # Test 1: Add chunks
        print("📝 Test 1: Adding chunks...")
        test_chunks = [
            CodeChunk(
                content="def hello_world():\n    print('Hello, World!')",
                file_path=Path("test.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                function_name="hello_world",
            ),
            CodeChunk(
                content="class Calculator:\n    def add(self, a, b):\n        return a + b",
                file_path=Path("test.py"),
                start_line=4,
                end_line=6,
                language="python",
                chunk_type="class",
                class_name="Calculator",
            ),
            CodeChunk(
                content="async def fetch_data(url):\n    return await http.get(url)",
                file_path=Path("async_test.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                function_name="fetch_data",
            ),
        ]

        await db.add_chunks(test_chunks)
        print(f"✅ Added {len(test_chunks)} chunks\n")

        # Test 2: Get stats
        print("📊 Test 2: Getting statistics...")
        stats = await db.get_stats()
        print(f"  Total chunks: {stats.total_chunks}")
        print(f"  Total files: {stats.total_files}")
        print(f"  Languages: {stats.languages}")
        print(f"  File types: {stats.file_types}")
        print(f"  Index size (MB): {stats.index_size_mb}")
        print(f"  Last updated: {stats.last_updated}")
        print(f"  Embedding model: {stats.embedding_model}")

        # Check all required fields are present
        assert stats.total_chunks == 3, f"Expected 3 chunks, got {stats.total_chunks}"
        assert stats.total_files == 2, f"Expected 2 files, got {stats.total_files}"
        assert "python" in stats.languages, (
            f"Expected 'python' in languages, got {stats.languages}"
        )
        assert ".py" in stats.file_types, (
            f"Expected '.py' in file_types, got {stats.file_types}"
        )
        assert stats.index_size_mb >= 0, (
            f"Expected non-negative index size, got {stats.index_size_mb}"
        )
        assert stats.last_updated is not None, "Expected last_updated to be set"
        assert stats.embedding_model is not None, "Expected embedding_model to be set"
        print("✅ Stats correct (all required fields present)\n")

        # Test 3: Search
        print("🔍 Test 3: Searching...")
        results = await db.search(
            query="function that prints hello world",
            limit=5,
            similarity_threshold=0.3,
        )
        print(f"  Found {len(results)} results")
        if results:
            print(
                f"  Top result: {results[0].function_name} (score: {results[0].similarity_score:.3f})"
            )
            assert results[0].function_name == "hello_world", (
                f"Expected 'hello_world', got {results[0].function_name}"
            )
        print("✅ Search works\n")

        # Test 4: Search with filters
        print("🔍 Test 4: Searching with file filter...")
        results = await db.search(
            query="async function",
            limit=5,
            filters={"file_path": "async_test.py"},
            similarity_threshold=0.3,
        )
        print(f"  Found {len(results)} results in async_test.py")
        if results:
            print(f"  Result: {results[0].function_name}")
            assert "async_test.py" in str(results[0].file_path), (
                f"Expected async_test.py, got {results[0].file_path}"
            )
        print("✅ Filtered search works\n")

        # Test 5: Get all chunks
        print("📥 Test 5: Getting all chunks...")
        all_chunks = await db.get_all_chunks()
        print(f"  Retrieved {len(all_chunks)} chunks")
        assert len(all_chunks) == 3, f"Expected 3 chunks, got {len(all_chunks)}"
        print("✅ Get all chunks works\n")

        # Test 6: Delete by file
        print("🗑️  Test 6: Deleting chunks by file...")
        deleted = await db.delete_by_file(Path("test.py"))
        print(f"  Deleted {deleted} chunks")
        assert deleted == 2, f"Expected to delete 2 chunks, deleted {deleted}"

        stats = await db.get_stats()
        print(f"  Remaining chunks: {stats.total_chunks}")
        assert stats.total_chunks == 1, (
            f"Expected 1 chunk remaining, got {stats.total_chunks}"
        )
        print("✅ Delete by file works\n")

        # Test 7: Health check
        print("🏥 Test 7: Health check...")
        healthy = await db.health_check()
        print(f"  Database healthy: {healthy}")
        assert healthy, "Database health check failed"
        print("✅ Health check works\n")

        # Test 7b: Health check auto-initialization
        print("🏥 Test 7b: Health check with uninitialized database...")
        # Create a new database instance without initializing
        db2 = LanceVectorDatabase(
            persist_directory=test_dir,
            embedding_function=embedding_function,
            collection_name="test_collection",
        )
        # Health check should auto-initialize
        healthy = await db2.health_check()
        print(f"  Database auto-initialized and healthy: {healthy}")
        assert healthy, "Database health check failed after auto-initialization"
        # Verify database is actually initialized
        assert db2._db is not None, "Database should be initialized after health check"
        print("✅ Health check auto-initialization works\n")

        # Test 8: Reset
        print("♻️  Test 8: Resetting database...")
        await db.reset()
        stats = await db.get_stats()
        print(f"  Chunks after reset: {stats.total_chunks}")
        assert stats.total_chunks == 0, (
            f"Expected 0 chunks after reset, got {stats.total_chunks}"
        )
        print("✅ Reset works\n")

    print("🎉 All tests passed!")

    # Cleanup
    import shutil

    shutil.rmtree(test_dir)
    print(f"🧹 Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    asyncio.run(test_lancedb_basic_operations())
