"""Tests for vectors backend (Phase 2 storage)."""

from pathlib import Path

import pytest

from src.mcp_vector_search.core.exceptions import (
    DatabaseError,
    DatabaseNotInitializedError,
    SearchError,
)
from src.mcp_vector_search.core.vectors_backend import VectorsBackend


@pytest.fixture
async def vectors_backend(tmp_path: Path):
    """Create and initialize vectors backend for testing."""
    backend = VectorsBackend(tmp_path / "test_db")
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def sample_chunks_with_vectors():
    """Sample chunks with vectors for testing."""
    return [
        {
            "chunk_id": "chunk1",
            "vector": [0.1] * 384,  # 384D vector
            "file_path": "src/main.py",
            "content": "def foo():\n    return 42",
            "language": "python",
            "start_line": 1,
            "end_line": 2,
            "chunk_type": "function",
            "name": "foo",
            "hierarchy_path": "foo",
        },
        {
            "chunk_id": "chunk2",
            "vector": [0.2] * 384,
            "file_path": "src/main.py",
            "content": "def bar():\n    return 'hello'",
            "language": "python",
            "start_line": 4,
            "end_line": 5,
            "chunk_type": "function",
            "name": "bar",
            "hierarchy_path": "bar",
        },
        {
            "chunk_id": "chunk3",
            "vector": [0.3] * 384,
            "file_path": "src/utils.js",
            "content": "function baz() {\n  return true;\n}",
            "language": "javascript",
            "start_line": 1,
            "end_line": 3,
            "chunk_type": "function",
            "name": "baz",
            "hierarchy_path": "baz",
        },
    ]


@pytest.mark.asyncio
class TestVectorsBackendInitialization:
    """Test vectors backend initialization."""

    async def test_initialize_creates_directory(self, tmp_path: Path):
        """Test that initialization creates database directory."""
        db_path = tmp_path / "test_db"
        assert not db_path.exists()

        backend = VectorsBackend(db_path)
        await backend.initialize()

        assert db_path.exists()
        assert db_path.is_dir()
        await backend.close()

    async def test_initialize_opens_existing_table(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that initialization opens existing table."""
        # Add some vectors
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        # Get the db_path before closing
        db_path = vectors_backend.db_path

        # Close and reinitialize
        await vectors_backend.close()

        # Create new backend instance with same path
        new_backend = VectorsBackend(db_path)
        await new_backend.initialize()

        # Should be able to access existing vectors
        stats = await new_backend.get_stats()
        assert stats["total"] == 3

        await new_backend.close()

    async def test_initialize_handles_missing_table(self, tmp_path: Path):
        """Test that initialization handles missing table gracefully."""
        backend = VectorsBackend(tmp_path / "new_db")
        await backend.initialize()

        # Table should be None until first add_vectors
        assert backend._table is None

        # Should return empty stats
        stats = await backend.get_stats()
        assert stats["total"] == 0
        await backend.close()


@pytest.mark.asyncio
class TestAddVectors:
    """Test adding vectors to table."""

    async def test_add_vectors_creates_table(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that add_vectors creates table on first call."""
        assert vectors_backend._table is None

        count = await vectors_backend.add_vectors(sample_chunks_with_vectors)

        assert count == 3
        assert vectors_backend._table is not None

    async def test_add_vectors_returns_count(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that add_vectors returns correct count."""
        count = await vectors_backend.add_vectors(sample_chunks_with_vectors)
        assert count == 3

    async def test_add_vectors_handles_empty_list(self, vectors_backend):
        """Test that add_vectors handles empty list gracefully."""
        count = await vectors_backend.add_vectors([])
        assert count == 0

    async def test_add_vectors_validates_chunk_id(self, vectors_backend):
        """Test that add_vectors validates chunk_id field."""
        invalid_chunk = {
            "vector": [0.1] * 384,
            "file_path": "src/main.py",
            "content": "def foo(): pass",
            "language": "python",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "function",
            "name": "foo",
        }

        with pytest.raises(DatabaseError) as exc_info:
            await vectors_backend.add_vectors([invalid_chunk])
        assert "chunk_id" in str(exc_info.value)

    async def test_add_vectors_validates_vector_field(self, vectors_backend):
        """Test that add_vectors validates vector field."""
        invalid_chunk = {
            "chunk_id": "chunk1",
            "file_path": "src/main.py",
            "content": "def foo(): pass",
            "language": "python",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "function",
            "name": "foo",
        }

        with pytest.raises(DatabaseError) as exc_info:
            await vectors_backend.add_vectors([invalid_chunk])
        assert "vector" in str(exc_info.value).lower()

    # NOTE: Dimension validation test removed - backend now auto-detects dimension
    # from first chunk, making this test obsolete. Auto-detection is the intended
    # behavior, not validation.

    async def test_add_vectors_sets_metadata(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that add_vectors sets embedded_at and model_version."""
        await vectors_backend.add_vectors(
            sample_chunks_with_vectors, model_version="test-model"
        )

        # Get vectors and check metadata
        df = vectors_backend._table.to_pandas()
        assert all(df["model_version"] == "test-model")
        assert all(df["embedded_at"].notna())

    async def test_add_vectors_requires_initialization(
        self, tmp_path: Path, sample_chunks_with_vectors
    ):
        """Test that add_vectors requires initialization."""
        backend = VectorsBackend(tmp_path / "test_db")
        # Don't initialize

        with pytest.raises(DatabaseNotInitializedError):
            await backend.add_vectors(sample_chunks_with_vectors)


@pytest.mark.asyncio
class TestVectorSearch:
    """Test vector similarity search."""

    async def test_search_returns_results(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search returns matching results."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        # Search with query vector similar to chunk1
        query_vector = [0.1] * 384
        results = await vectors_backend.search(query_vector, limit=10)

        assert len(results) == 3
        # With cosine similarity, order depends on vector normalization
        # Just verify we get results with reasonable similarity scores
        assert all(
            0.0 <= r["similarity"] <= 1.1 for r in results
        )  # Allow slight FP error
        chunk_ids = {r["chunk_id"] for r in results}
        assert chunk_ids == {"chunk1", "chunk2", "chunk3"}

    async def test_search_respects_limit(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search respects limit parameter."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search(query_vector, limit=2)

        assert len(results) == 2

    async def test_search_with_language_filter(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search filters by language."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search(
            query_vector, limit=10, filters={"language": "python"}
        )

        assert len(results) == 2
        assert all(r["language"] == "python" for r in results)

    async def test_search_with_file_path_filter(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search filters by file_path."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search(
            query_vector, limit=10, filters={"file_path": "src/main.py"}
        )

        assert len(results) == 2
        assert all(r["file_path"] == "src/main.py" for r in results)

    async def test_search_with_chunk_type_filter(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search filters by chunk_type."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search(
            query_vector, limit=10, filters={"chunk_type": "function"}
        )

        assert len(results) == 3
        assert all(r["chunk_type"] == "function" for r in results)

    async def test_search_with_multiple_filters(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search applies multiple filters."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search(
            query_vector,
            limit=10,
            filters={"language": "python", "file_path": "src/main.py"},
        )

        assert len(results) == 2
        assert all(r["language"] == "python" for r in results)
        assert all(r["file_path"] == "src/main.py" for r in results)

    async def test_search_returns_empty_for_empty_table(self, vectors_backend):
        """Test that search raises SearchError when table doesn't exist."""
        query_vector = [0.1] * 384

        from src.mcp_vector_search.core.exceptions import SearchError

        with pytest.raises(SearchError) as exc_info:
            await vectors_backend.search(query_vector, limit=10)
        assert "not found" in str(exc_info.value).lower()

    async def test_search_validates_vector_dimensions(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search validates query vector dimensions."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        invalid_query = [0.1] * 128  # Wrong dimension
        with pytest.raises(SearchError) as exc_info:
            await vectors_backend.search(invalid_query, limit=10)
        assert "dimension" in str(exc_info.value).lower()

    async def test_search_includes_similarity_score(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search results include similarity score."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search(query_vector, limit=10)

        assert len(results) > 0, "Should return results"
        for result in results:
            assert "similarity" in result, f"Result missing similarity: {result}"
            # Allow small floating point errors (cosine similarity can be slightly > 1.0)
            assert 0.0 <= result["similarity"] <= 1.01, (
                f"Invalid similarity: {result['similarity']}"
            )
            assert "_distance" in result, f"Result missing _distance: {result}"


@pytest.mark.asyncio
class TestSearchByFile:
    """Test file-specific search."""

    async def test_search_by_file_filters_results(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that search_by_file only returns results from specified file."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        query_vector = [0.1] * 384
        results = await vectors_backend.search_by_file(
            "src/main.py", query_vector, limit=10
        )

        assert len(results) == 2
        assert all(r["file_path"] == "src/main.py" for r in results)


@pytest.mark.asyncio
class TestDeleteFileVectors:
    """Test deleting file vectors."""

    async def test_delete_file_vectors_removes_all(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that delete_file_vectors removes all vectors for file."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        count = await vectors_backend.delete_file_vectors("src/main.py")

        assert count == 2

        # Verify vectors are deleted
        stats = await vectors_backend.get_stats()
        assert stats["total"] == 1  # Only chunk3 remains

    async def test_delete_file_vectors_returns_zero_for_missing_file(
        self, vectors_backend
    ):
        """Test that delete_file_vectors returns 0 for missing file."""
        count = await vectors_backend.delete_file_vectors("nonexistent.py")
        assert count == 0

    async def test_delete_file_vectors_preserves_other_files(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that delete_file_vectors doesn't affect other files."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        await vectors_backend.delete_file_vectors("src/main.py")

        # Verify only utils.js remains
        stats = await vectors_backend.get_stats()
        assert stats["total"] == 1
        assert stats["languages"]["javascript"] == 1


@pytest.mark.asyncio
class TestGetChunkVector:
    """Test retrieving chunk vectors."""

    async def test_get_chunk_vector_returns_vector(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that get_chunk_vector returns correct vector."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        vector = await vectors_backend.get_chunk_vector("chunk1")

        assert vector is not None
        assert len(vector) == 384
        # Use approximate comparison for floats (account for float32 precision)
        assert all(abs(v - 0.1) < 0.01 for v in vector)

    async def test_get_chunk_vector_returns_none_for_missing(self, vectors_backend):
        """Test that get_chunk_vector returns None for missing chunk."""
        vector = await vectors_backend.get_chunk_vector("nonexistent")
        assert vector is None


@pytest.mark.asyncio
class TestHasVector:
    """Test checking vector existence."""

    async def test_has_vector_returns_true_for_existing(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that has_vector returns True for existing chunk."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        has_it = await vectors_backend.has_vector("chunk1")
        assert has_it is True

    async def test_has_vector_returns_false_for_missing(self, vectors_backend):
        """Test that has_vector returns False for missing chunk."""
        has_it = await vectors_backend.has_vector("nonexistent")
        assert has_it is False


@pytest.mark.asyncio
class TestGetStats:
    """Test vector statistics."""

    async def test_get_stats_returns_correct_counts(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that get_stats returns correct counts."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        stats = await vectors_backend.get_stats()

        assert stats["total"] == 3
        assert stats["files"] == 2
        assert stats["languages"]["python"] == 2
        assert stats["languages"]["javascript"] == 1
        assert stats["chunk_types"]["function"] == 3

    async def test_get_stats_includes_model_versions(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that get_stats includes model version counts."""
        await vectors_backend.add_vectors(
            sample_chunks_with_vectors, model_version="test-model"
        )

        stats = await vectors_backend.get_stats()

        assert "models" in stats
        assert stats["models"]["test-model"] == 3

    async def test_get_stats_returns_empty_for_empty_table(self, vectors_backend):
        """Test that get_stats returns empty stats for empty table."""
        stats = await vectors_backend.get_stats()

        assert stats["total"] == 0
        assert stats["files"] == 0
        assert stats["languages"] == {}
        assert stats["chunk_types"] == {}


@pytest.mark.asyncio
class TestGetUnembeddedChunkIds:
    """Test finding unembedded chunks."""

    async def test_get_unembedded_finds_missing_chunks(self, vectors_backend):
        """Test that get_unembedded_chunk_ids finds missing chunks."""
        # Add vectors for some chunks
        chunks = [
            {
                "chunk_id": "chunk1",
                "vector": [0.1] * 384,
                "file_path": "src/main.py",
                "content": "def foo(): pass",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "function",
                "name": "foo",
                "hierarchy_path": "foo",
            }
        ]
        await vectors_backend.add_vectors(chunks)

        # Check which chunks need embedding
        all_chunk_ids = ["chunk1", "chunk2", "chunk3"]
        unembedded = await vectors_backend.get_unembedded_chunk_ids(all_chunk_ids)

        assert len(unembedded) == 2
        assert "chunk2" in unembedded
        assert "chunk3" in unembedded
        assert "chunk1" not in unembedded

    async def test_get_unembedded_returns_all_for_empty_table(self, vectors_backend):
        """Test that get_unembedded_chunk_ids returns all chunks for empty table."""
        all_chunk_ids = ["chunk1", "chunk2", "chunk3"]
        unembedded = await vectors_backend.get_unembedded_chunk_ids(all_chunk_ids)

        assert unembedded == all_chunk_ids

    async def test_get_unembedded_returns_empty_when_all_embedded(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that get_unembedded_chunk_ids returns empty when all embedded."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        all_chunk_ids = ["chunk1", "chunk2", "chunk3"]
        unembedded = await vectors_backend.get_unembedded_chunk_ids(all_chunk_ids)

        assert unembedded == []


@pytest.mark.asyncio
class TestRebuildIndex:
    """Test index rebuilding."""

    async def test_rebuild_index_succeeds(
        self, vectors_backend, sample_chunks_with_vectors
    ):
        """Test that rebuild_index completes successfully."""
        await vectors_backend.add_vectors(sample_chunks_with_vectors)

        # Should not raise an exception
        await vectors_backend.rebuild_index()

    async def test_rebuild_index_handles_empty_table(self, vectors_backend):
        """Test that rebuild_index handles empty table gracefully."""
        # Should not raise an exception
        await vectors_backend.rebuild_index()


@pytest.mark.asyncio
class TestContextManager:
    """Test async context manager support."""

    async def test_context_manager_initializes_and_closes(self, tmp_path: Path):
        """Test that context manager initializes and closes properly."""
        db_path = tmp_path / "test_db"

        async with VectorsBackend(db_path) as backend:
            assert backend._db is not None
            await backend.add_vectors(
                [
                    {
                        "chunk_id": "chunk1",
                        "vector": [0.1] * 384,
                        "file_path": "src/main.py",
                        "content": "def foo(): pass",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 1,
                        "chunk_type": "function",
                        "name": "foo",
                        "hierarchy_path": "foo",
                    }
                ]
            )

        # After exit, references should be None
        assert backend._db is None
        assert backend._table is None
