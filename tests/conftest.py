"""Pytest configuration and shared fixtures."""

import asyncio
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

# Import core modules for testing
from mcp_vector_search.core.database import VectorDatabase
from mcp_vector_search.core.embeddings import create_embedding_function

# from mcp_vector_search.core.factory import ComponentFactory, ComponentBundle  # TODO: Implement factory
from mcp_vector_search.core.models import CodeChunk, IndexStats, SearchResult

# Configure pytest-asyncio - asyncio_mode is set in pytest.ini


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def temp_project_dir(temp_dir: Path) -> Path:
    """Create a temporary project directory with sample files."""
    project_dir = temp_dir / "test_project"
    project_dir.mkdir()

    # Create sample Python files
    sample_files = {
        "main.py": '''
def main():
    """Main application entry point."""
    print("Hello, World!")
    user_service = UserService()
    users = user_service.get_all_users()
    return len(users)

if __name__ == "__main__":
    main()
''',
        "user_service.py": '''
from typing import List, Optional

class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email

    def __repr__(self):
        return f"User(id={self.id}, name='{self.name}', email='{self.email}')"

class UserService:
    def __init__(self):
        self.users = []

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user_id = len(self.users) + 1
        user = User(user_id, name, email)
        self.users.append(user)
        return user

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        for user in self.users:
            if user.id == user_id:
                return user
        return None

    def get_all_users(self) -> List[User]:
        """Get all users."""
        return self.users.copy()

    def delete_user(self, user_id: int) -> bool:
        """Delete user by ID."""
        for i, user in enumerate(self.users):
            if user.id == user_id:
                del self.users[i]
                return True
        return False
''',
        "utils.py": '''
import json
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save configuration to JSON file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)
''',
        "database.py": '''
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path

class DatabaseConnection:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row

    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT query."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query."""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor.rowcount

    def create_table(self, table_name: str, schema: str) -> None:
        """Create a table with the given schema."""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.execute_update(query)
''',
    }

    for filename, content in sample_files.items():
        (project_dir / filename).write_text(content)

    return project_dir


@pytest.fixture
def sample_code_chunks() -> list[CodeChunk]:
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            content="def main():\n    print('Hello, World!')",
            file_path=Path("main.py"),
            start_line=1,
            end_line=2,
            language="python",
            chunk_type="function",
            function_name="main",
            class_name=None,
        ),
        CodeChunk(
            content="class UserService:\n    def __init__(self):\n        self.users = []",
            file_path=Path("user_service.py"),
            start_line=10,
            end_line=12,
            language="python",
            chunk_type="class",
            function_name=None,
            class_name="UserService",
        ),
        CodeChunk(
            content="def create_user(self, name: str, email: str) -> User:",
            file_path=Path("user_service.py"),
            start_line=15,
            end_line=15,
            language="python",
            chunk_type="method",
            function_name="create_user",
            class_name="UserService",
        ),
    ]


@pytest.fixture
def mock_embedding_function():
    """Create a mock embedding function for testing."""

    class MockEmbeddingFunction:
        """ChromaDB-compatible mock embedding function."""

        def __init__(self):
            self._name = "test-embedding-function"

        def __call__(self, input: list[str]) -> list[list[float]]:
            """Generate deterministic mock embeddings."""
            embeddings = []
            for text in input:
                # Create a simple hash-based embedding (384-dimensional for all-MiniLM-L6-v2)
                embedding = [
                    float(hash(text + str(i)) % 100) / 100.0 for i in range(384)
                ]
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

    return MockEmbeddingFunction()


@pytest_asyncio.fixture
async def mock_database() -> AsyncGenerator[VectorDatabase, None]:
    """Create a mock vector database for testing."""

    class MockVectorDatabase(VectorDatabase):
        def __init__(self):
            self.chunks: list[CodeChunk] = []
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

        async def close(self) -> None:
            self.initialized = False

        async def add_chunks(
            self, chunks: list[CodeChunk], metrics: dict[str, Any] | None = None
        ) -> None:
            self.chunks.extend(chunks)
            # Store metrics if provided (for testing purposes)
            if metrics:
                self._metrics = metrics

        async def search(
            self,
            query: str,
            limit: int = 10,
            filters: dict[str, Any] = None,
            similarity_threshold: float = 0.7,
        ) -> list[SearchResult]:
            # Simple mock search based on text matching
            results = []
            for chunk in self.chunks:
                # Simple similarity calculation - normalize content for comparison
                query_words = set(query.lower().split())
                # Handle multi-line content by replacing newlines with spaces
                normalized_content = (
                    chunk.content.lower()
                    .replace("\n", " ")
                    .replace("(", " ")
                    .replace(")", " ")
                    .replace(":", " ")
                )
                content_words = set(normalized_content.split())

                if query_words.intersection(content_words):
                    similarity = len(query_words.intersection(content_words)) / len(
                        query_words.union(content_words)
                    )
                    if similarity >= similarity_threshold:
                        results.append(
                            SearchResult(
                                content=chunk.content,
                                file_path=chunk.file_path,
                                start_line=chunk.start_line,
                                end_line=chunk.end_line,
                                language=chunk.language,
                                similarity_score=similarity,
                                rank=len(results) + 1,
                                chunk_type=chunk.chunk_type,
                                function_name=chunk.function_name,
                                class_name=chunk.class_name,
                                context_before=[],
                                context_after=[],
                                highlights=[],
                            )
                        )

            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]

        async def delete_by_file(self, file_path: Path) -> int:
            initial_count = len(self.chunks)
            self.chunks = [c for c in self.chunks if c.file_path != file_path]
            return initial_count - len(self.chunks)

        async def get_stats(self) -> IndexStats:
            # Count languages and file types
            language_counts = {}
            file_types = {}
            files = set()

            for chunk in self.chunks:
                files.add(chunk.file_path)
                # Count languages
                if chunk.language:
                    language_counts[chunk.language] = (
                        language_counts.get(chunk.language, 0) + 1
                    )
                # Count file types by extension
                ext = chunk.file_path.suffix
                if ext:
                    file_types[ext] = file_types.get(ext, 0) + 1

            return IndexStats(
                total_chunks=len(self.chunks),
                total_files=len(files),
                languages=language_counts,
                file_types=file_types,
                index_size_mb=len(self.chunks) * 0.001,  # Mock size
                last_updated="2024-01-01T00:00:00",  # Mock timestamp
                embedding_model="test-embedding-model",  # Mock model name
            )

        async def get_all_chunks(self) -> list[CodeChunk]:
            """Get all chunks from the database."""
            return self.chunks.copy()

        async def health_check(self) -> bool:
            return self.initialized

        async def reset(self) -> None:
            """Reset the database."""
            self.chunks = []
            self.initialized = False

    db = MockVectorDatabase()
    yield db


@pytest.fixture
def real_embedding_function():
    """Create a real embedding function for integration tests."""
    embedding_function, _ = create_embedding_function(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_function


# TODO: Implement ComponentBundle and ComponentFactory
# @pytest.fixture
# async def components_bundle(temp_project_dir: Path, mock_database: VectorDatabase) -> ComponentBundle:
#     """Create a complete component bundle for testing."""
#     # Initialize project
#     project_manager = ProjectManager(temp_project_dir)
#     config = project_manager.initialize(
#         file_extensions=[".py"],
#         embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#         similarity_threshold=0.5,
#         force=True,
#     )
#
#     # Create mock embedding function
#     def mock_embeddings(texts: List[str]) -> List[List[float]]:
#         return [[0.1] * 384 for _ in texts]
#
#     # Create components
#     indexer = SemanticIndexer(
#         database=mock_database,
#         project_root=temp_project_dir,
#         file_extensions=[".py"],
#     )
#
#     search_engine = SemanticSearchEngine(
#         database=mock_database,
#         project_root=temp_project_dir,
#         similarity_threshold=0.1,
#     )
#
#     return ComponentBundle(
#         project_manager=project_manager,
#         config=config,
#         database=mock_database,
#         indexer=indexer,
#         embedding_function=mock_embeddings,
#         search_engine=search_engine,
#     )


# Performance testing utilities
class PerformanceTimer:
    """Utility for measuring performance in tests."""

    def __init__(self):
        self.times = []

    async def time_async_operation(self, operation, *args, **kwargs):
        """Time an async operation."""
        import time

        start = time.perf_counter()
        result = await operation(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        self.times.append(elapsed)
        return result, elapsed

    def get_stats(self):
        """Get performance statistics."""
        if not self.times:
            return {}

        import statistics

        return {
            "count": len(self.times),
            "total": sum(self.times),
            "average": statistics.mean(self.times),
            "median": statistics.median(self.times),
            "min": min(self.times),
            "max": max(self.times),
            "stdev": statistics.stdev(self.times) if len(self.times) > 1 else 0,
        }


@pytest.fixture
def performance_timer():
    """Create a performance timer for tests."""
    return PerformanceTimer()


# Test data generators
def generate_test_files(base_dir: Path, count: int = 10) -> list[Path]:
    """Generate test files for performance testing."""
    files = []
    for i in range(count):
        file_path = base_dir / f"test_file_{i}.py"
        content = f'''
def function_{i}():
    """Function number {i}."""
    return {i}

class Class{i}:
    def method_{i}(self):
        return "method_{i}"

    def process_data_{i}(self, data):
        """Process data for class {i}."""
        return data * {i}
'''
        file_path.write_text(content)
        files.append(file_path)

    return files


# CLI testing utilities
@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing commands."""
    from typer.testing import CliRunner

    return CliRunner()


# Assertion helpers
def assert_search_results_valid(results: list[SearchResult], min_count: int = 0):
    """Assert that search results are valid."""
    assert len(results) >= min_count, (
        f"Expected at least {min_count} results, got {len(results)}"
    )

    for result in results:
        assert isinstance(result, SearchResult)
        assert result.content is not None
        assert 0.0 <= result.similarity_score <= 1.0
        assert result.file_path is not None
        assert result.start_line >= 0
        assert result.end_line >= result.start_line


def assert_chunks_valid(chunks: list[CodeChunk]):
    """Assert that code chunks are valid."""
    assert len(chunks) > 0, "Expected at least one chunk"

    for chunk in chunks:
        assert isinstance(chunk, CodeChunk)
        assert chunk.id is not None
        assert chunk.content is not None
        assert chunk.file_path is not None
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line
        assert chunk.language is not None
