"""Tests for ChromaDB metadata schema extensions for structural code metrics."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_vector_search.analysis.metrics import ChunkMetrics
from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.models import CodeChunk


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """Create temporary database directory."""
    db_dir = tmp_path / "test_chromadb"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


class MockEmbeddingFunction:
    """Mock embedding function that implements ChromaDB's interface."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        return [[0.1, 0.2, 0.3]] * len(input)

    def name(self) -> str:
        """Return embedding function name."""
        return "mock_embeddings"


@pytest.fixture
def mock_embedding_function() -> MockEmbeddingFunction:
    """Create mock embedding function."""
    return MockEmbeddingFunction()


@pytest.fixture
def sample_chunk() -> CodeChunk:
    """Create sample code chunk."""
    return CodeChunk(
        content="def example():\n    return 42",
        file_path=Path("test.py"),
        start_line=1,
        end_line=2,
        language="python",
        chunk_type="function",
        function_name="example",
        chunk_id="test_chunk_123",
    )


@pytest.fixture
def sample_metrics() -> ChunkMetrics:
    """Create sample metrics."""
    return ChunkMetrics(
        cognitive_complexity=5,
        cyclomatic_complexity=2,
        max_nesting_depth=1,
        parameter_count=0,
        lines_of_code=2,
        smells=["too_short"],
    )


class TestDatabaseMetricsSupport:
    """Test ChromaDB metadata schema extensions for metrics."""

    @pytest.mark.asyncio
    async def test_add_chunks_with_metrics(
        self,
        temp_db_dir: Path,
        mock_embedding_function: MockEmbeddingFunction,
        sample_chunk: CodeChunk,
        sample_metrics: ChunkMetrics,
    ) -> None:
        """Test adding chunks with structural metrics."""
        db = ChromaVectorDatabase(
            persist_directory=temp_db_dir,
            embedding_function=mock_embedding_function,
        )

        await db.initialize()

        # Add chunk with metrics
        metrics_dict = {sample_chunk.chunk_id: sample_metrics.to_metadata()}
        await db.add_chunks([sample_chunk], metrics=metrics_dict)

        # Retrieve and verify metrics were stored
        chunks = await db.get_all_chunks()
        assert len(chunks) == 1

        # Verify metrics are in ChromaDB metadata (fetch directly from collection)
        results = db._collection.get(ids=[sample_chunk.id], include=["metadatas"])
        assert results["ids"]
        metadata = results["metadatas"][0]

        # Check all metrics fields
        assert metadata["cognitive_complexity"] == 5
        assert metadata["cyclomatic_complexity"] == 2
        assert metadata["max_nesting_depth"] == 1
        assert metadata["parameter_count"] == 0
        assert metadata["lines_of_code"] == 2
        assert metadata["complexity_grade"] == "A"  # Cognitive complexity 5 → Grade A
        assert metadata["code_smells"] == '["too_short"]'  # JSON string
        assert metadata["smell_count"] == 1

        await db.close()

    @pytest.mark.asyncio
    async def test_add_chunks_without_metrics(
        self,
        temp_db_dir: Path,
        mock_embedding_function: MockEmbeddingFunction,
        sample_chunk: CodeChunk,
    ) -> None:
        """Test backward compatibility: adding chunks without metrics."""
        db = ChromaVectorDatabase(
            persist_directory=temp_db_dir,
            embedding_function=mock_embedding_function,
        )

        await db.initialize()

        # Add chunk WITHOUT metrics (should work fine)
        await db.add_chunks([sample_chunk])

        # Retrieve chunk
        chunks = await db.get_all_chunks()
        assert len(chunks) == 1
        assert chunks[0].content == sample_chunk.content

        # Verify no metrics fields in metadata (they won't be present)
        results = db._collection.get(ids=[sample_chunk.id], include=["metadatas"])
        metadata = results["metadatas"][0]

        # Metrics fields should not be present
        assert "cognitive_complexity" not in metadata
        assert "complexity_grade" not in metadata
        assert "code_smells" not in metadata

        await db.close()

    @pytest.mark.asyncio
    async def test_search_filter_by_complexity_grade(
        self,
        temp_db_dir: Path,
        mock_embedding_function: MockEmbeddingFunction,
    ) -> None:
        """Test filtering search results by complexity grade."""
        db = ChromaVectorDatabase(
            persist_directory=temp_db_dir,
            embedding_function=mock_embedding_function,
        )

        await db.initialize()

        # Create chunks with different complexity grades
        chunks = [
            CodeChunk(
                content=f"def func_{i}():\n    return {i}",
                file_path=Path(f"test_{i}.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                function_name=f"func_{i}",
                chunk_id=f"chunk_{i}",
            )
            for i in range(3)
        ]

        # Create metrics with different grades
        metrics_dict = {
            "chunk_0": ChunkMetrics(cognitive_complexity=3).to_metadata(),  # Grade A
            "chunk_1": ChunkMetrics(cognitive_complexity=8).to_metadata(),  # Grade B
            "chunk_2": ChunkMetrics(cognitive_complexity=15).to_metadata(),  # Grade C
        }

        await db.add_chunks(chunks, metrics=metrics_dict)

        # Search with complexity_grade filter for grade A
        results = await db.search(
            query="function",
            limit=10,
            filters={"complexity_grade": "A"},
            similarity_threshold=0.0,
        )

        # Should only return grade A chunks
        assert len(results) == 1

        # Verify it's the grade A chunk by checking directly in ChromaDB
        all_results = db._collection.get(
            where={"complexity_grade": "A"}, include=["metadatas"]
        )
        assert len(all_results["ids"]) == 1
        assert all_results["metadatas"][0]["complexity_grade"] == "A"

        await db.close()

    @pytest.mark.asyncio
    async def test_search_filter_by_smell_count(
        self,
        temp_db_dir: Path,
        mock_embedding_function: MockEmbeddingFunction,
    ) -> None:
        """Test filtering by code smell count."""
        db = ChromaVectorDatabase(
            persist_directory=temp_db_dir,
            embedding_function=mock_embedding_function,
        )

        await db.initialize()

        # Create chunks with different smell counts
        chunks = [
            CodeChunk(
                content=f"def func_{i}():\n    return {i}",
                file_path=Path(f"test_{i}.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                function_name=f"func_{i}",
                chunk_id=f"chunk_{i}",
            )
            for i in range(3)
        ]

        metrics_dict = {
            "chunk_0": ChunkMetrics(smells=[]).to_metadata(),  # 0 smells
            "chunk_1": ChunkMetrics(smells=["too_complex"]).to_metadata(),  # 1 smell
            "chunk_2": ChunkMetrics(
                smells=["too_complex", "too_long"]
            ).to_metadata(),  # 2 smells
        }

        await db.add_chunks(chunks, metrics=metrics_dict)

        # Search for chunks with NO smells (smell_count = 0)
        results = await db.search(
            query="function",
            limit=10,
            filters={"smell_count": 0},
            similarity_threshold=0.0,
        )

        assert len(results) == 1

        # Verify it's the clean chunk
        clean_results = db._collection.get(
            where={"smell_count": 0}, include=["metadatas"]
        )
        assert len(clean_results["ids"]) == 1
        assert clean_results["metadatas"][0]["smell_count"] == 0

        # Search for chunks with smells (smell_count > 0)
        results = await db.search(
            query="function",
            limit=10,
            filters={"smell_count": {"$gt": 0}},
            similarity_threshold=0.0,
        )

        assert len(results) == 2  # Should return chunks with 1 and 2 smells

        await db.close()

    @pytest.mark.asyncio
    async def test_search_filter_by_complexity_range(
        self,
        temp_db_dir: Path,
        mock_embedding_function: MockEmbeddingFunction,
    ) -> None:
        """Test range queries on cognitive complexity."""
        db = ChromaVectorDatabase(
            persist_directory=temp_db_dir,
            embedding_function=mock_embedding_function,
        )

        await db.initialize()

        # Create chunks with different complexities
        chunks = [
            CodeChunk(
                content=f"def func_{i}():\n    return {i}",
                file_path=Path(f"test_{i}.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                function_name=f"func_{i}",
                chunk_id=f"chunk_{i}",
            )
            for i in range(4)
        ]

        metrics_dict = {
            "chunk_0": ChunkMetrics(cognitive_complexity=5).to_metadata(),
            "chunk_1": ChunkMetrics(cognitive_complexity=10).to_metadata(),
            "chunk_2": ChunkMetrics(cognitive_complexity=15).to_metadata(),
            "chunk_3": ChunkMetrics(cognitive_complexity=20).to_metadata(),
        }

        await db.add_chunks(chunks, metrics=metrics_dict)

        # Search for moderate complexity (10-20 range)
        # ChromaDB requires $and for range queries
        results = await db.search(
            query="function",
            limit=10,
            filters={
                "$and": [
                    {"cognitive_complexity": {"$gte": 10}},
                    {"cognitive_complexity": {"$lte": 20}},
                ]
            },
            similarity_threshold=0.0,
        )

        # Should return chunks with complexity 10, 15, 20
        assert len(results) == 3

        await db.close()

    @pytest.mark.asyncio
    async def test_multiple_chunks_same_file(
        self,
        temp_db_dir: Path,
        mock_embedding_function: MockEmbeddingFunction,
    ) -> None:
        """Test adding multiple chunks with different metrics from same file."""
        db = ChromaVectorDatabase(
            persist_directory=temp_db_dir,
            embedding_function=mock_embedding_function,
        )

        await db.initialize()

        # Create multiple chunks from same file
        chunks = [
            CodeChunk(
                content="def func_a():\n    return 1",
                file_path=Path("test.py"),
                start_line=1,
                end_line=2,
                language="python",
                chunk_type="function",
                function_name="func_a",
                chunk_id="chunk_a",
            ),
            CodeChunk(
                content="def func_b():\n    return 2",
                file_path=Path("test.py"),
                start_line=4,
                end_line=5,
                language="python",
                chunk_type="function",
                function_name="func_b",
                chunk_id="chunk_b",
            ),
        ]

        # Different metrics for each chunk
        metrics_dict = {
            "chunk_a": ChunkMetrics(cognitive_complexity=3).to_metadata(),  # Grade A
            "chunk_b": ChunkMetrics(cognitive_complexity=25).to_metadata(),  # Grade D
        }

        await db.add_chunks(chunks, metrics=metrics_dict)

        # Verify both chunks stored with correct metrics
        all_chunks = await db.get_all_chunks()
        assert len(all_chunks) == 2

        # Verify metrics in database
        for chunk in chunks:
            results = db._collection.get(ids=[chunk.id], include=["metadatas"])
            metadata = results["metadatas"][0]
            assert "cognitive_complexity" in metadata
            assert "complexity_grade" in metadata

        await db.close()


class TestMigrationScript:
    """Test migration script functionality (mock-based)."""

    @pytest.mark.asyncio
    async def test_migration_adds_default_metrics(self, tmp_path: Path) -> None:
        """Test that migration adds default metrics to chunks without them."""
        # Create temporary directory for test
        test_dir = tmp_path / "chromadb"
        test_dir.mkdir()

        # Mock ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "metadatas": [
                {"file_path": "test1.py", "language": "python"},  # No metrics
                {
                    "file_path": "test2.py",
                    "language": "python",
                    "cognitive_complexity": 5,
                },  # Has metrics
            ],
        }
        mock_collection.update = MagicMock()

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        # Mock chromadb.PersistentClient
        with patch("chromadb.PersistentClient", return_value=mock_client):
            from scripts.migrate_chromadb_metrics import migrate_metrics

            stats = await migrate_metrics(
                persist_dir=test_dir, dry_run=False, batch_size=10
            )

            # Should migrate 1 chunk (chunk1), skip 1 chunk (chunk2)
            assert stats["total"] == 2
            assert stats["migrated"] == 1
            assert stats["skipped"] == 1
            assert stats["errors"] == 0

            # Verify update was called with correct default metrics
            mock_collection.update.assert_called_once()
            call_args = mock_collection.update.call_args
            assert call_args.kwargs["ids"] == ["chunk1"]

            updated_metadata = call_args.kwargs["metadatas"][0]
            assert updated_metadata["cognitive_complexity"] == 0
            assert updated_metadata["cyclomatic_complexity"] == 1
            assert updated_metadata["complexity_grade"] == "A"
            assert updated_metadata["code_smells"] == "[]"  # JSON string
            assert updated_metadata["smell_count"] == 0

    @pytest.mark.asyncio
    async def test_migration_dry_run(self, tmp_path: Path) -> None:
        """Test migration in dry-run mode doesn't modify database."""
        # Create temporary directory for test
        test_dir = tmp_path / "chromadb"
        test_dir.mkdir()

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.get.return_value = {
            "ids": ["chunk1"],
            "metadatas": [{"file_path": "test1.py"}],  # No metrics
        }
        mock_collection.update = MagicMock()

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client):
            from scripts.migrate_chromadb_metrics import migrate_metrics

            stats = await migrate_metrics(
                persist_dir=test_dir, dry_run=True, batch_size=10
            )

            # Should report migration but not actually update
            assert stats["migrated"] == 1
            mock_collection.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_migration_empty_collection(self) -> None:
        """Test migration handles empty collection gracefully."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client):
            from scripts.migrate_chromadb_metrics import migrate_metrics

            stats = await migrate_metrics(
                persist_dir=Path("/tmp/test"), dry_run=False, batch_size=10
            )

            assert stats["total"] == 0
            assert stats["migrated"] == 0
            assert stats["skipped"] == 0


class TestMetricsIntegration:
    """Integration tests for metrics with ChunkMetrics dataclass."""

    def test_chunk_metrics_to_metadata_format(self) -> None:
        """Test ChunkMetrics.to_metadata() produces ChromaDB-compatible format."""
        metrics = ChunkMetrics(
            cognitive_complexity=15,
            cyclomatic_complexity=5,
            max_nesting_depth=3,
            parameter_count=4,
            lines_of_code=50,
            smells=["too_complex", "too_many_params"],
        )

        metadata = metrics.to_metadata()

        # Verify all required fields
        assert metadata["cognitive_complexity"] == 15
        assert metadata["cyclomatic_complexity"] == 5
        assert metadata["max_nesting_depth"] == 3
        assert metadata["parameter_count"] == 4
        assert metadata["lines_of_code"] == 50
        assert metadata["complexity_grade"] == "C"  # 15 → grade C
        assert metadata["smell_count"] == 2

        # Verify types are ChromaDB-compatible
        assert isinstance(metadata["cognitive_complexity"], int)
        assert isinstance(metadata["complexity_grade"], str)
        assert isinstance(metadata["code_smells"], str)  # JSON string now
        # Verify we can parse it back to a list
        import json

        parsed_smells = json.loads(metadata["code_smells"])
        assert parsed_smells == ["too_complex", "too_many_params"]

    def test_complexity_grade_calculation(self) -> None:
        """Test complexity grade is correctly calculated for different values."""
        test_cases = [
            (0, "A"),
            (5, "A"),
            (6, "B"),
            (10, "B"),
            (11, "C"),
            (20, "C"),
            (21, "D"),
            (30, "D"),
            (31, "F"),
            (100, "F"),
        ]

        for complexity, expected_grade in test_cases:
            metrics = ChunkMetrics(cognitive_complexity=complexity)
            assert metrics.complexity_grade == expected_grade, (
                f"Complexity {complexity} should be grade {expected_grade}"
            )
