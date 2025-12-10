"""Unit tests for core models."""

from pathlib import Path

from mcp_vector_search.core.models import CodeChunk, IndexStats, SearchResult


class TestCodeChunk:
    """Test cases for CodeChunk model."""

    def test_code_chunk_creation(self):
        """Test basic CodeChunk creation."""
        chunk = CodeChunk(
            content="def hello():\n    print('Hello, World!')",
            file_path=Path("test.py"),
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            function_name="hello",
        )

        assert chunk.content == "def hello():\n    print('Hello, World!')"
        assert chunk.file_path == Path("test.py")
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.language == "python"
        assert chunk.chunk_type == "function"
        assert chunk.function_name == "hello"

    def test_code_chunk_id_property(self):
        """Test CodeChunk ID generation."""
        chunk = CodeChunk(
            content="test content",
            file_path=Path("test.py"),
            start_line=10,
            end_line=15,
            language="python",
        )

        expected_id = "test.py:10:15"
        assert chunk.id == expected_id

    def test_code_chunk_line_count(self):
        """Test line count calculation."""
        chunk = CodeChunk(
            content="line1\nline2\nline3",
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            language="python",
        )

        assert chunk.line_count == 3

    def test_code_chunk_to_dict(self):
        """Test CodeChunk serialization."""
        chunk = CodeChunk(
            content="def test(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            chunk_type="function",
            function_name="test",
        )

        chunk_dict = chunk.to_dict()

        assert chunk_dict["content"] == "def test(): pass"
        assert chunk_dict["file_path"] == "test.py"
        assert chunk_dict["start_line"] == 1
        assert chunk_dict["end_line"] == 1
        assert chunk_dict["language"] == "python"
        assert chunk_dict["chunk_type"] == "function"
        assert chunk_dict["function_name"] == "test"

    def test_code_chunk_from_dict(self):
        """Test CodeChunk deserialization."""
        data = {
            "content": "def test(): pass",
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "chunk_type": "function",
            "function_name": "test",
        }

        chunk = CodeChunk.from_dict(data)

        assert chunk.content == "def test(): pass"
        assert chunk.file_path == Path("test.py")
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.language == "python"
        assert chunk.chunk_type == "function"
        assert chunk.function_name == "test"


class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_search_result_creation(self):
        """Test basic SearchResult creation."""
        result = SearchResult(
            content="def hello(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.85,
            rank=1,
            chunk_type="function",
            function_name="hello",
        )

        assert result.content == "def hello(): pass"
        assert result.file_path == Path("test.py")
        assert result.start_line == 1
        assert result.end_line == 1
        assert result.language == "python"
        assert result.similarity_score == 0.85
        assert result.rank == 1
        assert result.chunk_type == "function"
        assert result.function_name == "hello"

    def test_search_result_line_count(self):
        """Test line count calculation."""
        result = SearchResult(
            content="line1\nline2\nline3",
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            language="python",
            similarity_score=0.8,
            rank=1,
        )

        assert result.line_count == 3

    def test_search_result_location(self):
        """Test location string generation."""
        result = SearchResult(
            content="test content",
            file_path=Path("test.py"),
            start_line=10,
            end_line=15,
            language="python",
            similarity_score=0.8,
            rank=1,
        )

        assert result.location == "test.py:10-15"

    def test_search_result_to_dict(self):
        """Test SearchResult serialization."""
        result = SearchResult(
            content="def test(): pass",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            similarity_score=0.9,
            rank=1,
            chunk_type="function",
            function_name="test",
            context_before=["# Comment"],
            context_after=["# End"],
            highlights=["test"],
        )

        result_dict = result.to_dict()

        assert result_dict["content"] == "def test(): pass"
        assert result_dict["file_path"] == "test.py"
        assert result_dict["start_line"] == 1
        assert result_dict["end_line"] == 1
        assert result_dict["language"] == "python"
        assert result_dict["similarity_score"] == 0.9
        assert result_dict["rank"] == 1
        assert result_dict["chunk_type"] == "function"
        assert result_dict["function_name"] == "test"
        assert result_dict["context_before"] == ["# Comment"]
        assert result_dict["context_after"] == ["# End"]
        assert result_dict["highlights"] == ["test"]


class TestIndexStats:
    """Test cases for IndexStats model."""

    def test_index_stats_creation(self):
        """Test basic IndexStats creation."""
        stats = IndexStats(
            total_files=10,
            total_chunks=100,
            languages={"python": 80, "javascript": 20},
            file_types={".py": 8, ".js": 2},
            index_size_mb=5.2,
            last_updated="2024-01-01T00:00:00Z",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert stats.total_chunks == 100
        assert stats.total_files == 10
        assert stats.languages == {"python": 80, "javascript": 20}
        assert stats.file_types == {".py": 8, ".js": 2}
        assert stats.index_size_mb == 5.2
        assert stats.last_updated == "2024-01-01T00:00:00Z"
        assert stats.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_index_stats_to_dict(self):
        """Test IndexStats serialization."""
        stats = IndexStats(
            total_files=5,
            total_chunks=50,
            languages={"python": 50},
            file_types={".py": 5},
            index_size_mb=2.1,
            last_updated="2024-01-01T00:00:00Z",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

        stats_dict = stats.to_dict()

        assert stats_dict["total_chunks"] == 50
        assert stats_dict["total_files"] == 5
        assert stats_dict["languages"] == {"python": 50}
        assert stats_dict["file_types"] == {".py": 5}
        assert stats_dict["index_size_mb"] == 2.1
        assert stats_dict["last_updated"] == "2024-01-01T00:00:00Z"
        assert stats_dict["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
