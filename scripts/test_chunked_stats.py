#!/usr/bin/env python3
"""Test script to verify chunked statistics processing with large datasets."""

import asyncio
import tempfile
from pathlib import Path

from mcp_vector_search.core.database import (
    ChromaVectorDatabase,
    PooledChromaVectorDatabase,
)
from mcp_vector_search.core.models import CodeChunk


class MockEmbeddingFunction:
    """Mock embedding function for testing."""

    def __init__(self):
        self._name = "test-embedding-function"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        embeddings = []
        for text in input:
            # Generate consistent but unique embeddings based on text hash
            embedding = [float(hash(text + str(i)) % 100) / 100.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings

    def name(self) -> str:
        """Return the name of the embedding function."""
        return self._name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self.__call__(input=texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.__call__(input=[text])[0]


async def create_large_dataset(count: int) -> list[CodeChunk]:
    """Create a large dataset of code chunks for testing."""
    chunks = []

    for i in range(count):
        chunk = CodeChunk(
            content=f"def function_{i}():\n    pass",
            file_path=Path(f"file_{i % 100}.py"),
            start_line=i * 10,
            end_line=i * 10 + 2,
            language="python",
            chunk_type="function",
            function_name=f"function_{i}",
        )
        chunks.append(chunk)

    return chunks


async def test_regular_database(count: int):
    """Test ChromaVectorDatabase with chunked stats."""
    print(f"\n🔹 Testing ChromaVectorDatabase with {count} chunks...")

    with tempfile.TemporaryDirectory() as temp_dir:
        db = ChromaVectorDatabase(
            persist_directory=Path(temp_dir) / "regular_db",
            embedding_function=MockEmbeddingFunction(),
            collection_name="test_collection",
        )

        try:
            await db.initialize()

            # Create and add chunks
            print(f"  Creating {count} chunks...")
            chunks = await create_large_dataset(count)

            print("  Adding chunks to database...")
            await db.add_chunks(chunks)

            # Get stats (should use chunked processing)
            print("  Fetching stats with chunked processing...")
            stats = await db.get_stats()

            # Verify stats
            print("\n  ✅ Stats retrieved successfully:")
            print(f"     Total chunks: {stats.total_chunks}")
            print(f"     Total files: {stats.total_files}")
            print(f"     Languages: {stats.languages}")
            print(f"     Index size: {stats.index_size_mb:.2f} MB")

            assert stats.total_chunks == count, (
                f"Expected {count} chunks, got {stats.total_chunks}"
            )
            assert stats.total_files > 0, "Expected files to be counted"

            print("\n  ✅ All assertions passed!")

        finally:
            await db.close()


async def test_pooled_database(count: int):
    """Test PooledChromaVectorDatabase with chunked stats."""
    print(f"\n🔹 Testing PooledChromaVectorDatabase with {count} chunks...")

    with tempfile.TemporaryDirectory() as temp_dir:
        db = PooledChromaVectorDatabase(
            persist_directory=Path(temp_dir) / "pooled_db",
            embedding_function=MockEmbeddingFunction(),
            collection_name="test_collection",
            max_connections=5,
            min_connections=2,
        )

        try:
            await db.initialize()

            # Create and add chunks
            print(f"  Creating {count} chunks...")
            chunks = await create_large_dataset(count)

            print("  Adding chunks to database...")
            await db.add_chunks(chunks)

            # Get stats (should use chunked processing)
            print("  Fetching stats with chunked processing...")
            stats = await db.get_stats()

            # Verify stats
            print("\n  ✅ Stats retrieved successfully:")
            print(f"     Total chunks: {stats.total_chunks}")
            print(f"     Total files: {stats.total_files}")
            print(f"     Languages: {stats.languages}")
            print(f"     Index size: {stats.index_size_mb:.2f} MB")

            # Check pool stats
            pool_stats = db.get_pool_stats()
            print("\n  📊 Pool statistics:")
            print(f"     Pool size: {pool_stats['pool_size']}")
            print(f"     Connections created: {pool_stats['connections_created']}")
            print(f"     Connections reused: {pool_stats['connections_reused']}")

            assert stats.total_chunks == count, (
                f"Expected {count} chunks, got {stats.total_chunks}"
            )
            assert stats.total_files > 0, "Expected files to be counted"

            print("\n  ✅ All assertions passed!")

        finally:
            await db.close()


async def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Chunked Database Statistics Processing")
    print("=" * 80)

    # Test with different dataset sizes
    test_sizes = [100, 1000, 5000]

    for size in test_sizes:
        print(f"\n{'=' * 80}")
        print(f"Dataset Size: {size} chunks")
        print("=" * 80)

        try:
            await test_regular_database(size)
            await test_pooled_database(size)
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback

            traceback.print_exc()
            return 1

    print("\n" + "=" * 80)
    print("✅ All tests passed successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
