"""Tests for two-phase indexing architecture (Phase 1: Chunk, Phase 2: Embed)."""

import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.chunks_backend import compute_file_hash
from mcp_vector_search.core.database import VectorDatabase
from mcp_vector_search.core.embeddings import CodeEmbeddings
from mcp_vector_search.core.indexer import SemanticIndexer


@pytest.fixture
async def test_project():
    """Create a temporary test project with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create sample Python file
        (project_root / "test.py").write_text(
            """
def hello():
    '''Say hello'''
    return "Hello, world!"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
"""
        )

        # Create sample JavaScript file
        (project_root / "test.js").write_text(
            """
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}
"""
        )

        yield project_root


@pytest.mark.asyncio
async def test_phase1_chunk_only(test_project):
    """Test Phase 1: Chunk files without embedding."""
    # Initialize database and indexer
    embeddings = CodeEmbeddings()
    database = VectorDatabase(test_project / ".mcp-vector-search", embeddings)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=test_project,
        file_extensions=[".py", ".js"],
    )

    # Run Phase 1 only
    indexed_count = await indexer.index_project(phase="chunk")

    # Verify files were chunked
    assert indexed_count == 2  # test.py and test.js

    # Verify chunks backend has data
    chunks_backend = indexer.chunks_backend
    stats = await chunks_backend.get_stats()

    assert stats["total"] > 0  # Has chunks
    assert stats["pending"] > 0  # Chunks are pending embedding
    assert stats["complete"] == 0  # No chunks embedded yet

    await database.close()


@pytest.mark.asyncio
async def test_phase2_embed_only(test_project):
    """Test Phase 2: Embed pending chunks."""
    # Initialize database and indexer
    embeddings = CodeEmbeddings()
    database = VectorDatabase(test_project / ".mcp-vector-search", embeddings)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=test_project,
        file_extensions=[".py", ".js"],
    )

    # Run Phase 1 first
    await indexer.index_project(phase="chunk")

    # Get stats before Phase 2
    chunks_stats_before = await indexer.chunks_backend.get_stats()
    pending_before = chunks_stats_before["pending"]

    # Run Phase 2 only
    await indexer.index_project(phase="embed")

    # Verify embeddings were generated
    chunks_stats_after = await indexer.chunks_backend.get_stats()
    vectors_stats = await indexer.vectors_backend.get_stats()

    # Pending chunks should decrease
    assert chunks_stats_after["pending"] <= pending_before
    # Complete chunks should increase
    assert chunks_stats_after["complete"] > 0
    # Vectors should be created
    assert vectors_stats["total"] > 0

    await database.close()


@pytest.mark.asyncio
async def test_two_phase_full_workflow(test_project):
    """Test full two-phase workflow (both phases)."""
    # Initialize database and indexer
    embeddings = CodeEmbeddings()
    database = VectorDatabase(test_project / ".mcp-vector-search", embeddings)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=test_project,
        file_extensions=[".py", ".js"],
    )

    # Run both phases (default behavior)
    indexed_count = await indexer.index_project(phase="all")

    # Verify both phases completed
    assert indexed_count == 2

    # Get final stats
    status = await indexer.get_two_phase_status()

    # Phase 1 should have chunks
    assert status["phase1"]["total_chunks"] > 0
    assert status["phase1"]["files_indexed"] == 2

    # Phase 2 should show completion
    assert status["phase2"]["complete"] > 0

    # Vectors should be ready
    assert status["vectors"]["total"] > 0
    assert status["ready_for_search"] is True

    await database.close()


@pytest.mark.asyncio
async def test_incremental_update_via_file_hash(test_project):
    """Test incremental updates using file hash change detection."""
    # Initialize database and indexer
    embeddings = CodeEmbeddings()
    database = VectorDatabase(test_project / ".mcp-vector-search", embeddings)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=test_project,
        file_extensions=[".py", ".js"],
    )

    # Initial index
    await indexer.index_project()

    # Get initial hash
    test_py = test_project / "test.py"
    initial_hash = compute_file_hash(test_py)

    # Modify file
    test_py.write_text(
        """
def hello():
    '''Say hello'''
    return "Hello, modified world!"
"""
    )

    # Verify hash changed
    new_hash = compute_file_hash(test_py)
    assert new_hash != initial_hash

    # Run incremental index (should only reindex changed file)
    indexed_count = await indexer.index_project(force_reindex=False)

    # Should have re-indexed the changed file
    assert indexed_count == 1

    await database.close()


@pytest.mark.asyncio
async def test_crash_recovery_resume_embedding(test_project):
    """Test crash recovery: resume embedding after interruption."""
    # Initialize database and indexer
    embeddings = CodeEmbeddings()
    database = VectorDatabase(test_project / ".mcp-vector-search", embeddings)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=test_project,
        file_extensions=[".py", ".js"],
    )

    # Run Phase 1 only
    await indexer.index_project(phase="chunk")

    # Simulate crash by marking some chunks as "processing"
    chunks_backend = indexer.chunks_backend
    pending = await chunks_backend.get_pending_chunks(limit=5)
    if pending:
        chunk_ids = [c["chunk_id"] for c in pending[:2]]
        await chunks_backend.mark_chunks_processing(chunk_ids, batch_id=12345)

    # Cleanup stale processing chunks (crash recovery)
    reset_count = await chunks_backend.cleanup_stale_processing(older_than_minutes=0)
    assert reset_count > 0  # Chunks were reset to pending

    # Now resume Phase 2
    await indexer.index_project(phase="embed")

    # Verify all chunks are now complete or error
    stats = await chunks_backend.get_stats()
    assert stats["processing"] == 0  # No stuck chunks
    assert stats["complete"] + stats["error"] == stats["total"]

    await database.close()


@pytest.mark.asyncio
async def test_get_two_phase_status(test_project):
    """Test status reporting for two-phase architecture."""
    # Initialize database and indexer
    embeddings = CodeEmbeddings()
    database = VectorDatabase(test_project / ".mcp-vector-search", embeddings)
    await database.initialize()

    indexer = SemanticIndexer(
        database=database,
        project_root=test_project,
        file_extensions=[".py", ".js"],
    )

    # Initial status (empty)
    status = await indexer.get_two_phase_status()
    assert status["phase1"]["total_chunks"] == 0
    assert status["ready_for_search"] is False

    # Run Phase 1
    await indexer.index_project(phase="chunk")

    # Status after Phase 1
    status = await indexer.get_two_phase_status()
    assert status["phase1"]["total_chunks"] > 0
    assert status["phase2"]["pending"] > 0
    assert status["phase2"]["complete"] == 0
    assert status["ready_for_search"] is False  # Not ready until embedded

    # Run Phase 2
    await indexer.index_project(phase="embed")

    # Status after Phase 2
    status = await indexer.get_two_phase_status()
    assert status["phase2"]["complete"] > 0
    assert status["vectors"]["total"] > 0
    assert status["ready_for_search"] is True  # Now ready for search

    await database.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
