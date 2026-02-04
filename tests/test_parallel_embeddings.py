"""Integration test for parallel embedding generation."""

import os
import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.embeddings import (
    BatchEmbeddingProcessor,
    CodeBERTEmbeddingFunction,
    EmbeddingCache,
)


@pytest.fixture
def embedding_function():
    """Create embedding function for testing."""
    return CodeBERTEmbeddingFunction(model_name="microsoft/codebert-base")


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_parallel_embedding_enabled(embedding_function, cache_dir):
    """Test that parallel embedding is enabled by default for large batches."""
    # Set environment variable to enable parallel embeddings
    os.environ["MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS"] = "true"

    cache = EmbeddingCache(cache_dir=cache_dir, max_size=100)
    processor = BatchEmbeddingProcessor(
        embedding_function=embedding_function, cache=cache, batch_size=16
    )

    # Create 64 test strings (above threshold for parallel processing)
    test_contents = [f"def function_{i}(): pass" for i in range(64)]

    # Process batch with parallel embeddings
    embeddings = await processor.process_batch(test_contents)

    # Verify we got embeddings for all inputs
    assert len(embeddings) == 64
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 768 for emb in embeddings)  # CodeBERT dimension


@pytest.mark.asyncio
async def test_parallel_embedding_disabled(embedding_function, cache_dir):
    """Test that parallel embedding can be disabled via environment variable."""
    # Disable parallel embeddings
    os.environ["MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS"] = "false"

    cache = EmbeddingCache(cache_dir=cache_dir, max_size=100)
    processor = BatchEmbeddingProcessor(
        embedding_function=embedding_function, cache=cache, batch_size=16
    )

    # Create 64 test strings
    test_contents = [f"def function_{i}(): pass" for i in range(64)]

    # Process batch with sequential embeddings
    embeddings = await processor.process_batch(test_contents)

    # Verify we got embeddings for all inputs
    assert len(embeddings) == 64
    assert all(isinstance(emb, list) for emb in embeddings)


@pytest.mark.asyncio
async def test_sequential_fallback_on_small_batch(embedding_function, cache_dir):
    """Test that small batches (<32) use sequential processing."""
    os.environ["MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS"] = "true"

    cache = EmbeddingCache(cache_dir=cache_dir, max_size=100)
    processor = BatchEmbeddingProcessor(
        embedding_function=embedding_function, cache=cache, batch_size=8
    )

    # Create small batch (below parallel threshold)
    test_contents = [f"def function_{i}(): pass" for i in range(16)]

    # Process batch - should use sequential
    embeddings = await processor.process_batch(test_contents)

    # Verify we got embeddings
    assert len(embeddings) == 16
    assert all(isinstance(emb, list) for emb in embeddings)


@pytest.mark.asyncio
async def test_cache_integration_with_parallel(embedding_function, cache_dir):
    """Test that caching works correctly with parallel embedding."""
    os.environ["MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS"] = "true"

    cache = EmbeddingCache(cache_dir=cache_dir, max_size=100)
    processor = BatchEmbeddingProcessor(
        embedding_function=embedding_function, cache=cache, batch_size=16
    )

    # First batch - no cache hits
    test_contents = [f"def function_{i}(): pass" for i in range(40)]
    embeddings1 = await processor.process_batch(test_contents)

    # Get cache stats
    stats1 = cache.get_cache_stats()
    assert stats1["cache_hits"] == 0
    assert stats1["cache_misses"] == 40

    # Second batch with same content - should have cache hits
    embeddings2 = await processor.process_batch(test_contents)

    # Get updated stats
    stats2 = cache.get_cache_stats()
    assert stats2["cache_hits"] == 40  # All should be cache hits
    assert stats2["cache_misses"] == 40  # Still same misses from first batch

    # Verify embeddings are identical
    assert embeddings1 == embeddings2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
