"""Tests for chunks backend (Phase 1 storage)."""

import tempfile
from pathlib import Path

import pytest

from mcp_vector_search.core.chunks_backend import (
    ChunksBackend,
    compute_file_hash,
)
from mcp_vector_search.core.exceptions import (
    DatabaseNotInitializedError,
)


@pytest.fixture
def temp_db_path():
    """Create temporary directory for test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": "chunk1",
            "file_path": "src/main.py",
            "content": "def hello():\n    print('Hello')",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": "hello",
            "signature": "def hello()",
            "complexity": 1,
            "token_count": 10,
        },
        {
            "chunk_id": "chunk2",
            "file_path": "src/main.py",
            "content": "class MyClass:\n    pass",
            "language": "python",
            "start_line": 4,
            "end_line": 5,
            "chunk_type": "class",
            "name": "MyClass",
            "signature": "class MyClass",
            "complexity": 1,
            "token_count": 8,
        },
    ]


@pytest.mark.asyncio
async def test_backend_initialization(temp_db_path):
    """Test backend initialization creates database."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    assert backend._db is not None
    assert temp_db_path.exists()

    await backend.close()


@pytest.mark.asyncio
async def test_add_chunks(temp_db_path, sample_chunks):
    """Test adding chunks to database."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    file_hash = "abc123"
    count = await backend.add_chunks(sample_chunks, file_hash)

    assert count == 2

    # Verify chunks were added
    stats = await backend.get_stats()
    assert stats["total"] == 2
    assert stats["pending"] == 2
    assert stats["files"] == 1

    await backend.close()


@pytest.mark.asyncio
async def test_add_chunks_not_initialized(temp_db_path, sample_chunks):
    """Test adding chunks without initialization raises error."""
    backend = ChunksBackend(temp_db_path)

    with pytest.raises(DatabaseNotInitializedError):
        await backend.add_chunks(sample_chunks, "hash")


@pytest.mark.asyncio
async def test_file_hash_tracking(temp_db_path, sample_chunks):
    """Test file hash tracking for change detection."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    file_hash = "abc123"
    await backend.add_chunks(sample_chunks, file_hash)

    # Get stored hash
    stored_hash = await backend.get_file_hash("src/main.py")
    assert stored_hash == file_hash

    # Check non-existent file
    missing_hash = await backend.get_file_hash("src/missing.py")
    assert missing_hash is None

    await backend.close()


@pytest.mark.asyncio
async def test_file_changed_detection(temp_db_path, sample_chunks):
    """Test file change detection logic."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    file_hash = "abc123"
    await backend.add_chunks(sample_chunks, file_hash)

    # Same hash = not changed
    changed = await backend.file_changed("src/main.py", "abc123")
    assert not changed

    # Different hash = changed
    changed = await backend.file_changed("src/main.py", "xyz789")
    assert changed

    # New file = changed
    changed = await backend.file_changed("src/new.py", "def456")
    assert changed

    await backend.close()


@pytest.mark.asyncio
async def test_get_pending_chunks(temp_db_path, sample_chunks):
    """Test retrieving pending chunks."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    file_hash = "abc123"
    await backend.add_chunks(sample_chunks, file_hash)

    # Get pending chunks
    pending = await backend.get_pending_chunks(batch_size=10)
    assert len(pending) == 2
    assert all(chunk["embedding_status"] == "pending" for chunk in pending)

    await backend.close()


@pytest.mark.asyncio
async def test_get_pending_chunks_pagination(temp_db_path):
    """Test pagination of pending chunks."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    # Add 5 chunks
    chunks = [
        {
            "chunk_id": f"chunk{i}",
            "file_path": "test.py",
            "content": f"code{i}",
            "language": "python",
            "start_line": i,
            "end_line": i + 1,
            "chunk_type": "function",
            "name": f"func{i}",
        }
        for i in range(5)
    ]
    await backend.add_chunks(chunks, "hash123")

    # Get first 2
    batch1 = await backend.get_pending_chunks(batch_size=2, offset=0)
    assert len(batch1) == 2

    # Get next 2
    batch2 = await backend.get_pending_chunks(batch_size=2, offset=2)
    assert len(batch2) == 2

    # Get last 1
    batch3 = await backend.get_pending_chunks(batch_size=2, offset=4)
    assert len(batch3) == 1

    await backend.close()


@pytest.mark.asyncio
async def test_mark_chunks_processing(temp_db_path, sample_chunks):
    """Test marking chunks as processing."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    await backend.add_chunks(sample_chunks, "hash123")

    # Mark as processing
    chunk_ids = ["chunk1", "chunk2"]
    batch_id = 42
    await backend.mark_chunks_processing(chunk_ids, batch_id)

    # Verify status changed
    stats = await backend.get_stats()
    assert stats["processing"] == 2
    assert stats["pending"] == 0

    await backend.close()


@pytest.mark.asyncio
async def test_mark_chunks_complete(temp_db_path, sample_chunks):
    """Test marking chunks as complete."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    await backend.add_chunks(sample_chunks, "hash123")

    # Mark as complete
    chunk_ids = ["chunk1", "chunk2"]
    await backend.mark_chunks_complete(chunk_ids)

    # Verify status changed
    stats = await backend.get_stats()
    assert stats["complete"] == 2
    assert stats["pending"] == 0

    await backend.close()


@pytest.mark.asyncio
async def test_mark_chunks_error(temp_db_path, sample_chunks):
    """Test marking chunks as error."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    await backend.add_chunks(sample_chunks, "hash123")

    # Mark as error
    chunk_ids = ["chunk1", "chunk2"]
    error_msg = "Embedding generation failed"
    await backend.mark_chunks_error(chunk_ids, error_msg)

    # Verify status changed
    stats = await backend.get_stats()
    assert stats["error"] == 2
    assert stats["pending"] == 0

    await backend.close()


@pytest.mark.asyncio
async def test_delete_file_chunks(temp_db_path):
    """Test deleting chunks for a specific file."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    # Add chunks for two files
    chunks_file1 = [
        {
            "chunk_id": "chunk1",
            "file_path": "file1.py",
            "content": "code1",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": "func1",
        },
        {
            "chunk_id": "chunk2",
            "file_path": "file1.py",
            "content": "code2",
            "language": "python",
            "start_line": 3,
            "end_line": 4,
            "chunk_type": "function",
            "name": "func2",
        },
    ]
    chunks_file2 = [
        {
            "chunk_id": "chunk3",
            "file_path": "file2.py",
            "content": "code3",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": "func3",
        }
    ]

    await backend.add_chunks(chunks_file1, "hash1")
    await backend.add_chunks(chunks_file2, "hash2")

    # Verify initial state
    stats = await backend.get_stats()
    assert stats["total"] == 3
    assert stats["files"] == 2

    # Delete file1 chunks
    deleted = await backend.delete_file_chunks("file1.py")
    assert deleted == 2

    # Verify file1 chunks removed
    stats = await backend.get_stats()
    assert stats["total"] == 1
    assert stats["files"] == 1

    await backend.close()


@pytest.mark.asyncio
async def test_get_stats(temp_db_path):
    """Test statistics generation."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    # Add chunks with different statuses
    chunks = [
        {
            "chunk_id": f"chunk{i}",
            "file_path": f"file{i // 2}.py",  # 2 chunks per file
            "content": f"code{i}",
            "language": "python" if i % 2 == 0 else "javascript",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": f"func{i}",
        }
        for i in range(10)
    ]
    await backend.add_chunks(chunks, "hash123")

    # Mark some as complete
    await backend.mark_chunks_complete(["chunk0", "chunk1"])

    # Mark some as processing
    await backend.mark_chunks_processing(["chunk2", "chunk3"], 1)

    # Mark some as error
    await backend.mark_chunks_error(["chunk4"], "Test error")

    # Get stats
    stats = await backend.get_stats()
    assert stats["total"] == 10
    assert stats["complete"] == 2
    assert stats["processing"] == 2
    assert stats["error"] == 1
    assert stats["pending"] == 5
    assert stats["files"] == 5
    assert "python" in stats["languages"]
    assert "javascript" in stats["languages"]

    await backend.close()


@pytest.mark.asyncio
async def test_cleanup_stale_processing(temp_db_path, sample_chunks):
    """Test cleanup of stale processing chunks."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    await backend.add_chunks(sample_chunks, "hash123")

    # Mark as processing
    await backend.mark_chunks_processing(["chunk1", "chunk2"], 1)

    # Verify processing status
    stats = await backend.get_stats()
    assert stats["processing"] == 2

    # Cleanup with short timeout (should not reset recent chunks)
    reset = await backend.cleanup_stale_processing(older_than_minutes=60)
    assert reset == 0

    # Verify still processing
    stats = await backend.get_stats()
    assert stats["processing"] == 2

    # Note: Testing actual time-based cleanup is difficult without
    # manipulating timestamps. In production, stale chunks would be
    # older than the cutoff.

    await backend.close()


@pytest.mark.asyncio
async def test_compute_file_hash():
    """Test file hash computation."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)

    try:
        hash1 = compute_file_hash(temp_path)
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex length

        # Same content = same hash
        hash2 = compute_file_hash(temp_path)
        assert hash1 == hash2

        # Different content = different hash
        temp_path.write_text("different content")
        hash3 = compute_file_hash(temp_path)
        assert hash1 != hash3

    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_context_manager(temp_db_path, sample_chunks):
    """Test async context manager usage."""
    async with ChunksBackend(temp_db_path) as backend:
        await backend.add_chunks(sample_chunks, "hash123")
        stats = await backend.get_stats()
        assert stats["total"] == 2

    # Backend should be closed after context
    # (LanceDB doesn't require explicit closing, but references should be None)
    assert backend._db is None
    assert backend._table is None


@pytest.mark.asyncio
async def test_empty_chunks_list(temp_db_path):
    """Test adding empty chunks list."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    count = await backend.add_chunks([], "hash123")
    assert count == 0

    stats = await backend.get_stats()
    assert stats["total"] == 0

    await backend.close()


@pytest.mark.asyncio
async def test_status_transitions(temp_db_path, sample_chunks):
    """Test complete status transition workflow."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    # Phase 1: Add chunks (pending)
    await backend.add_chunks(sample_chunks, "hash123")
    stats = await backend.get_stats()
    assert stats["pending"] == 2

    # Phase 2: Mark as processing
    chunk_ids = ["chunk1", "chunk2"]
    await backend.mark_chunks_processing(chunk_ids, batch_id=1)
    stats = await backend.get_stats()
    assert stats["processing"] == 2
    assert stats["pending"] == 0

    # Phase 3: Mark as complete
    await backend.mark_chunks_complete(chunk_ids)
    stats = await backend.get_stats()
    assert stats["complete"] == 2
    assert stats["processing"] == 0

    await backend.close()


@pytest.mark.asyncio
async def test_incremental_indexing_workflow(temp_db_path):
    """Test complete incremental indexing workflow."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()

    # Initial indexing
    file_path = "src/main.py"
    initial_hash = "hash_v1"
    chunks_v1 = [
        {
            "chunk_id": "chunk1",
            "file_path": file_path,
            "content": "old content",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": "func1",
        }
    ]
    await backend.add_chunks(chunks_v1, initial_hash)

    # Check if file changed (same hash)
    changed = await backend.file_changed(file_path, initial_hash)
    assert not changed  # No change, skip re-indexing

    # Simulate file modification
    new_hash = "hash_v2"
    changed = await backend.file_changed(file_path, new_hash)
    assert changed  # File changed, need re-indexing

    # Re-index: delete old chunks and add new
    deleted = await backend.delete_file_chunks(file_path)
    assert deleted == 1

    chunks_v2 = [
        {
            "chunk_id": "chunk2",
            "file_path": file_path,
            "content": "new content",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": "func2",
        }
    ]
    await backend.add_chunks(chunks_v2, new_hash)

    # Verify new chunks stored
    stored_hash = await backend.get_file_hash(file_path)
    assert stored_hash == new_hash

    stats = await backend.get_stats()
    assert stats["total"] == 1

    await backend.close()


@pytest.mark.asyncio
async def test_initialize_idempotent_no_table_exists_error(temp_db_path, sample_chunks):
    """Test that calling initialize() twice does NOT raise 'Table already exists'."""
    backend = ChunksBackend(temp_db_path)

    # First init then add chunks to create the table
    await backend.initialize()
    await backend.add_chunks(sample_chunks, "hash1")
    await backend.close()

    # Second init on the same path must open the existing table without error
    backend2 = ChunksBackend(temp_db_path)
    await backend2.initialize()  # Must NOT raise "Table already exists"
    stats = await backend2.get_stats()
    assert stats["total"] == 2
    await backend2.close()


@pytest.mark.asyncio
async def test_initialize_force_drops_existing_table(temp_db_path, sample_chunks):
    """Test that initialize(force=True) drops and recreates the existing table."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()
    await backend.add_chunks(sample_chunks, "hash1")
    await backend.close()

    # force=True must drop the table (no data after reopen)
    backend2 = ChunksBackend(temp_db_path)
    await backend2.initialize(force=True)
    # Table dropped — _table is None until first write
    assert backend2._table is None
    stats = await backend2.get_stats()
    assert stats["total"] == 0
    await backend2.close()


@pytest.mark.asyncio
async def test_initialize_force_false_preserves_existing_data(
    temp_db_path, sample_chunks
):
    """Test that initialize(force=False) preserves existing table data."""
    backend = ChunksBackend(temp_db_path)
    await backend.initialize()
    await backend.add_chunks(sample_chunks, "hash1")
    await backend.close()

    # force=False (default) must open existing table, not drop it
    backend2 = ChunksBackend(temp_db_path)
    await backend2.initialize(force=False)
    stats = await backend2.get_stats()
    assert stats["total"] == 2
    await backend2.close()
