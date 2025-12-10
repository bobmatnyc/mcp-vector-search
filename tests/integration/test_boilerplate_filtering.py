"""Integration tests for boilerplate filtering in semantic search."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_vector_search.core.models import SearchResult
from mcp_vector_search.core.search import SemanticSearchEngine


class TestBoilerplateFilteringIntegration:
    """Integration tests for boilerplate filtering in search results."""

    def _create_search_result(
        self,
        content: str,
        file_path: str,
        start_line: int,
        end_line: int,
        language: str,
        similarity_score: float,
        chunk_type: str = "function",
        function_name: str | None = None,
        class_name: str | None = None,
    ) -> SearchResult:
        """Helper to create SearchResult objects."""
        return SearchResult(
            content=content,
            file_path=Path(file_path),
            start_line=start_line,
            end_line=end_line,
            language=language,
            similarity_score=similarity_score,
            rank=0,
            chunk_type=chunk_type,
            function_name=function_name,
            class_name=class_name,
        )

    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        db = Mock()
        db.search = AsyncMock()
        db.health_check = AsyncMock(return_value=True)
        return db

    @pytest.fixture
    def search_engine(self, mock_database, tmp_path):
        """Create a search engine instance for testing."""
        # Patch file reading to avoid actual file I/O
        with patch.object(
            SemanticSearchEngine,
            "_read_file_lines_cached",
            new=AsyncMock(return_value=[]),
        ):
            engine = SemanticSearchEngine(
                database=mock_database,
                project_root=tmp_path,
                similarity_threshold=0.3,
            )
        return engine

    @pytest.mark.asyncio
    async def test_boilerplate_penalty_applied_to_init_method(
        self, search_engine, mock_database
    ):
        """Test that __init__ methods receive penalty in search results."""
        # Mock database results with __init__ and custom method
        mock_results = [
            self._create_search_result(
                content="def __init__(self, name):\n    self.name = name",
                file_path="/test/user.py",
                start_line=1,
                end_line=2,
                language="python",
                similarity_score=0.8,
                function_name="__init__",
                class_name="User",
            ),
            self._create_search_result(
                content="def process_user_data(self, data):\n    return data.upper()",
                file_path="/test/user.py",
                start_line=4,
                end_line=5,
                language="python",
                similarity_score=0.75,
                function_name="process_user_data",
                class_name="User",
            ),
        ]
        mock_database.search.return_value = mock_results

        # Patch file reading to avoid I/O
        with patch.object(
            search_engine, "_read_file_lines_cached", new=AsyncMock(return_value=[])
        ):
            results = await search_engine.search("user data processing", limit=10)

        # The custom method should rank higher than __init__ after reranking
        # because __init__ gets a -0.15 penalty
        assert len(results) == 2
        assert results[0].function_name == "process_user_data"
        assert results[1].function_name == "__init__"
        assert results[0].similarity_score > results[1].similarity_score

    @pytest.mark.asyncio
    async def test_explicit_init_query_bypasses_penalty(
        self, search_engine, mock_database
    ):
        """Test that searching for __init__ explicitly doesn't penalize it."""
        mock_results = [
            self._create_search_result(
                content="def __init__(self, name):\n    self.name = name",
                file_path="/test/user.py",
                start_line=1,
                end_line=2,
                language="python",
                similarity_score=0.9,
                function_name="__init__",
                class_name="User",
            ),
            self._create_search_result(
                content="def process_user_data(self, data):\n    return data",
                file_path="/test/user.py",
                start_line=4,
                end_line=5,
                language="python",
                similarity_score=0.7,
                function_name="process_user_data",
                class_name="User",
            ),
        ]
        mock_database.search.return_value = mock_results

        with patch.object(
            search_engine, "_read_file_lines_cached", new=AsyncMock(return_value=[])
        ):
            results = await search_engine.search("show __init__ methods", limit=10)

        # __init__ should maintain higher ranking because query contains "__init__"
        assert len(results) == 2
        assert results[0].function_name == "__init__"
        assert results[0].similarity_score > results[1].similarity_score

    @pytest.mark.asyncio
    async def test_javascript_constructor_penalty(self, search_engine, mock_database):
        """Test that JavaScript constructor methods receive penalty."""
        mock_results = [
            self._create_search_result(
                content="constructor(name) {\n  this.name = name;\n}",
                file_path="/test/User.js",
                start_line=2,
                end_line=4,
                language="javascript",
                similarity_score=0.8,
                function_name="constructor",
                class_name="User",
            ),
            self._create_search_result(
                content="authenticate(password) {\n  return validatePassword(password);\n}",
                file_path="/test/User.js",
                start_line=6,
                end_line=8,
                language="javascript",
                similarity_score=0.75,
                function_name="authenticate",
                class_name="User",
            ),
        ]
        mock_database.search.return_value = mock_results

        with patch.object(
            search_engine, "_read_file_lines_cached", new=AsyncMock(return_value=[])
        ):
            results = await search_engine.search("user authentication", limit=10)

        # Custom method should rank higher after constructor penalty
        assert len(results) == 2
        assert results[0].function_name == "authenticate"
        assert results[1].function_name == "constructor"

    @pytest.mark.asyncio
    async def test_mixed_languages_boilerplate_filtering(
        self, search_engine, mock_database
    ):
        """Test boilerplate filtering across multiple languages."""
        mock_results = [
            self._create_search_result(
                content="def __init__(self):\n    pass",
                file_path="/test/user.py",
                start_line=1,
                end_line=2,
                language="python",
                similarity_score=0.8,
                function_name="__init__",
            ),
            self._create_search_result(
                content="constructor() {}",
                file_path="/test/User.js",
                start_line=1,
                end_line=1,
                language="javascript",
                similarity_score=0.78,
                function_name="constructor",
            ),
            self._create_search_result(
                content="def process_data(data):\n    return data",
                file_path="/test/processor.py",
                start_line=1,
                end_line=2,
                language="python",
                similarity_score=0.7,
                function_name="process_data",
            ),
        ]
        mock_database.search.return_value = mock_results

        with patch.object(
            search_engine, "_read_file_lines_cached", new=AsyncMock(return_value=[])
        ):
            results = await search_engine.search("data processing", limit=10)

        # Custom method should rank highest
        assert len(results) == 3
        assert results[0].function_name == "process_data"
        # Boilerplate methods should be at the end
        boilerplate_names = {results[1].function_name, results[2].function_name}
        assert "__init__" in boilerplate_names
        assert "constructor" in boilerplate_names

    @pytest.mark.asyncio
    async def test_non_function_results_unaffected(self, search_engine, mock_database):
        """Test that non-function chunks are not affected by boilerplate filtering."""
        mock_results = [
            self._create_search_result(
                content="class User:\n    def __init__(self):\n        pass",
                file_path="/test/user.py",
                start_line=1,
                end_line=3,
                language="python",
                similarity_score=0.8,
                chunk_type="class",
                class_name="User",
                function_name=None,
            ),
            self._create_search_result(
                content="# Initialize the database connection",
                file_path="/test/db.py",
                start_line=1,
                end_line=1,
                language="python",
                similarity_score=0.7,
                chunk_type="comment",
                function_name=None,
            ),
        ]
        mock_database.search.return_value = mock_results

        with patch.object(
            search_engine, "_read_file_lines_cached", new=AsyncMock(return_value=[])
        ):
            results = await search_engine.search("database initialization", limit=10)

        # Results should maintain original order (no function_name penalty applied)
        assert len(results) == 2
        assert results[0].chunk_type == "class"
        assert results[1].chunk_type == "comment"
